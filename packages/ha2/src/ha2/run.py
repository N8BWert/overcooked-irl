"""Train, evaluate, and watch the same saved hierarchical policies."""

import argparse
import json
import importlib.metadata
import hashlib
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO

from ha2.world import World
from ha2.training import SelfPlay, Progress


def train(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "config.json"
    feature_version = 2
    recipe_mask = True
    if config_path.exists() and (args.resume or args.stage == "manager"):
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        feature_version = previous.get("feature_version", 1)
        recipe_mask = previous.get("menu_recipe_mask", feature_version >= 2)
        if (previous["layout"], previous["menu"]) != (args.layout, args.menu):
            raise ValueError(
                "Checkpoint layout/menu differs; use a new output directory."
            )
    if args.stage == "worker" and (output / "manager.zip").exists():
        raise ValueError(
            "Use a new output directory to retrain the worker; this manager requires its fixed worker."
        )
    worker = (
        PPO.load(output / "worker", device="cpu") if args.stage == "manager" else None
    )
    env = SelfPlay(
        args.stage,
        args.layout,
        args.menu,
        worker,
        args.seed,
        args.curriculum,
        feature_version,
        recipe_mask,
    )
    algorithm = PPO if args.stage == "worker" else MaskablePPO
    path = output / args.stage
    if args.resume:
        model = algorithm.load(path, env=env, device="cpu")
    else:
        model = algorithm(
            "MlpPolicy",
            env,
            n_steps=512,
            batch_size=256,
            n_epochs=4,
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            policy_kwargs={"net_arch": [64, 64]},
            seed=args.seed,
            device="cpu",
            verbose=0,
        )
    metadata = {
        "layout": args.layout,
        "menu": args.menu,
        "horizon": 900,
        "observation": "full-state structured features and path distances",
        "tasks": env.world.tasks,
        "shared_parameters": True,
        "fcp": False,
        "cogrid_version": importlib.metadata.version("cogrid"),
        "gamma": 0.99,
        "worker_inference": "sampled",
        "task_rules": "plate demand; wait only without productive tasks",
        "feature_version": feature_version,
        "menu_recipe_mask": recipe_mask,
    }
    (output / "config.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(
        f"Training {args.stage}: {args.steps} agent transitions, shared-policy self-play.",
        flush=True,
    )
    started = time.monotonic()
    model.learn(
        args.steps, callback=Progress(path), reset_num_timesteps=not args.resume
    )
    model.save(path)
    metadata["worker_sha256"] = hashlib.sha256(
        (output / "worker.zip").read_bytes()
    ).hexdigest()
    config_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    with (output / "training_history.jsonl").open("a", encoding="utf-8") as log:
        log.write(
            json.dumps(
                {
                    "stage": args.stage,
                    "seed": args.seed,
                    "curriculum": args.curriculum,
                    "menu_recipe_mask": recipe_mask,
                    "practice_orders": bool(args.curriculum and recipe_mask),
                    "resume": args.resume,
                    "model_steps": model.num_timesteps,
                    "seconds": time.monotonic() - started,
                }
            )
            + "\n"
        )
    print(f"Saved {path}.zip in {time.monotonic() - started:.1f}s", flush=True)
    env.close()


def evaluate(args):
    output = Path(args.output)
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    if (
        "worker_sha256" in config
        and hashlib.sha256((output / "worker.zip").read_bytes()).hexdigest()
        != config["worker_sha256"]
    ):
        raise ValueError("The worker checkpoint changed after training this manager.")
    worker = PPO.load(output / "worker", device="cpu")
    if args.worker_only:
        env = SelfPlay(
            "worker",
            config["layout"],
            config["menu"],
            seed=args.seed,
            feature_version=config.get("feature_version", 1),
            recipe_mask=config.get("menu_recipe_mask", False),
        )
        obs = env.reset()
        counts = {name: [0, 0] for name in env.world.tasks}
        for _ in range(args.steps // 2):
            actions = worker.predict(obs, deterministic=True)[0]
            obs, _, _, infos = env.step(actions)
            for info in infos:
                if "task" in info:
                    row = counts[env.world.tasks[info["task"]]]
                    row[0] += int(info["skill_success"])
                    row[1] += 1
        report = {
            "attempts": env.attempts,
            "successes": env.successes,
            "success_rate": env.successes / max(env.attempts, 1),
            "per_task_successes_trials": counts,
        }
        (output / f"worker_evaluation_{args.seed}.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(json.dumps(report), flush=True)
        env.close()
        return
    manager = MaskablePPO.load(output / "manager", device="cpu")
    w = World(
        config["layout"],
        config["menu"],
        config["horizon"],
        config.get("feature_version", 1),
        config.get("menu_recipe_mask", False),
    )
    reports = []
    screen = clock = None
    if args.render:
        import pygame
        from common.ui_utilities import draw_orders

        pygame.init()
        w.env.render_mode = "rgb_array"
        w.env._sync_objects_from_state()
        h, width = w.env.get_full_render().shape[:2]
        screen = pygame.display.set_mode((width * 2, h * 2 + 144))
        pygame.display.set_caption("HA2 Self-Play")
        clock = pygame.time.Clock()
    for seed in range(args.seed, args.seed + args.episodes):
        torch.manual_seed(seed)
        w.reset(seed)
        score, deliveries = 0.0, [0, 0]
        by_recipe = {r.result: 0 for r in w.training_recipes}
        incorrect = 0
        for step in range(config["horizon"]):
            inputs = np.stack([w.manager_obs(a) for a in range(2)])
            masks = np.stack([w.mask(a) for a in range(2)])
            tasks = manager.predict(
                inputs, deterministic=not args.stochastic, action_masks=masks
            )[0]
            primitive = worker.predict(
                np.stack([w.worker_obs(a, tasks[a]) for a in range(2)]),
                deterministic=args.deterministic_worker,
            )[0]
            primitive[tasks == w.wait] = 6
            previous = w.state
            goals = [w.targets(a, tasks[a]) for a in range(2)]
            _, rewards, terms, truncs, _ = w.step(primitive)
            score += rewards[0]
            available = list(previous.extra_state["overcooked.order_recipe"])
            for a in range(2):
                if tasks[a] == w.wait - 3 and w.completed(
                    a, tasks[a], previous, goals[a], primitive[a]
                ):
                    recipe = w.recipe_ids.index(int(previous.agent_inv[a, 0]))
                    if recipe in available:
                        available.remove(recipe)
                        deliveries[a] += 1
                        name = w.recipes[recipe].result
                        by_recipe[name] = by_recipe.get(name, 0) + 1
                    else:
                        incorrect += 1
            if args.render:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                image = w.env.get_full_render()
                surface = pygame.surfarray.make_surface(image.swapaxes(0, 1))
                screen.fill((24, 24, 24))
                screen.blit(pygame.transform.scale_by(surface, 2), (0, 144))
                draw_orders(
                    screen,
                    w.env,
                    [r.result for r in w.recipes],
                    time_limit=w.env.config["orders"]["time_limit"],
                )
                font = pygame.font.Font(None, 22)
                for a in range(2):
                    label = f"Player {a + 1}: {w.tasks[tasks[a]]} | Delivered: {deliveries[a]}"
                    screen.blit(
                        font.render(label, True, (245, 245, 245)), (8, 68 + a * 24)
                    )
                label = f"Seed: {seed} | Step: {step + 1} | Team return: {score:.0f}"
                screen.blit(font.render(label, True, (245, 245, 245)), (8, 116))
                pygame.display.flip()
                clock.tick(args.fps)
            if any(terms.values()) or any(truncs.values()):
                break
        report = {
            "seed": seed,
            "manager_inference": "sampled" if args.stochastic else "greedy",
            "worker_inference": "greedy" if args.deterministic_worker else "sampled",
            "steps": step + 1,
            "team_return": score,
            "deliveries_by_player": deliveries,
            "deliveries": sum(deliveries),
            "deliveries_by_recipe": by_recipe,
            "incorrect_deliveries": incorrect,
            "expired_orders": int(w.extra("order_n_expired")),
        }
        reports.append(report)
        print(json.dumps(report), flush=True)
    suffix = "_sampled" if args.stochastic else ""
    report_path = output / f"evaluation_{args.seed}{suffix}.json"
    report_path.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    w.env.close()
    if args.render:
        pygame.quit()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train", "evaluate"])
    parser.add_argument("--stage", choices=["worker", "manager"], default="worker")
    parser.add_argument("--layout", default="kitchen")
    parser.add_argument("--menu", default="onion")
    parser.add_argument("--output", default="checkpoints/ha2")
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--curriculum",
        action="store_true",
        help="Practice manager on randomized intermediate states",
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--worker-only", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument(
        "--stochastic", action="store_true", help="Also sample manager decisions"
    )
    parser.add_argument(
        "--deterministic-worker",
        action="store_true",
        help="Use argmax worker actions (may deadlock in narrow passages)",
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.command == "train":
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
