export PYTHONPATH=.
uv run ruff check --fix .
uv run black .
uv run isort .
uv run pytest -q
uv run pre-commit run --all-files
