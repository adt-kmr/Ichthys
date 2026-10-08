# Contributing to Ichthys

Thank you for considering contributing! Please see the following guidelines:

## Development Setup

1. Clone the repository
2. Create a virtual environment: `python -m venv .venv`
3. Activate: `.venv\Scripts\activate`
4. Install dependencies: `pip install -e .[viz]`
5. Run tests: `python -m pytest tests/`

## Adding New Features

1. Add new functionality to the appropriate module
2. Update docstrings and type hints
3. Add or update tests in `tests/`
4. Update the README if needed

## Running the Dashboard

```bash
streamlit run app.py -- --workspace /path/to/outputs
```