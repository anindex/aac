# Contributing to AAC

Thank you for your interest in contributing to AAC! This document provides
guidelines for contributing to this project.

## Getting Started

1. Fork the repository and clone your fork.
2. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,experiments]"
```

3. Run the test suite to make sure everything works:

```bash
pytest tests/ -x --tb=short
```

## Development Workflow

1. Create a feature branch from `main`.
2. Make your changes with clear, descriptive commit messages.
3. Add tests for new functionality.
4. Run the linter and type checker:

```bash
ruff check src/ scripts/ tests/ examples/
mypy src/aac/
```

5. Run the full test suite before submitting.
6. Open a pull request with a clear description of your changes.

## Code Style

- We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting.
- Line length: 100 characters.
- Type hints are expected for all public function signatures.
- Docstrings follow NumPy style.

## Testing

- Tests live in `tests/` and use `pytest`.
- Mark slow tests with `@pytest.mark.slow`.
- Mark GPU tests with `@pytest.mark.gpu`.
- Mark tests requiring network with `@pytest.mark.network`.

## Reporting Issues

Please use GitHub Issues to report bugs or request features. When reporting
a bug, include:

- Python version and OS
- PyTorch version
- Minimal reproducible example
- Full error traceback
