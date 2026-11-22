# Overcooked Interactive Robot Learning

## Description

This is a project attempting to utilize human feedback to improve the performance of an agent playing overcooked with a human.

## Developing

This project uses a justfile for organizing commands.  Right now the following commands exist:
* `just init`: Run this once to initialize the project dependencies
* `just format`: Run this once to use ruff to fix the code formatting for the project
* `just run-coop {{layout}} {{menu}}`: Run this to run two-player local overcooked using a specified menu and layout
