# Justfile for working with the overcooked-irl project

init:
    uv sync --all-packages
    uvx ruff check --select I --fix

format:
    uvx ruff check --select I --fix

run-coop layout="kitchen" menu="menu1":
    uv run human-cooperation --menu {{menu}} --layout {{layout}}
