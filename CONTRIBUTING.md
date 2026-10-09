# Contributing to Ichthys

Thank you for considering contributing to Ichthys! This project geometry-supervised 3D multi-object tracking for schooling fish.

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
4. Follow the existing code style and patterns
5. Add keyboard shortcuts and progress indicators where applicable

## UX Improvements

When adding new features, consider:
- **Error messages**: Provide helpful error messages with suggested fixes
- **Progress indicators**: Use `tqdm` for long-running operations
- **Command completion**: Add bash/zsh completion scripts
- **Keyboard shortcuts**: Document and implement dashboard shortcuts

## Reporting Issues

If you encounter bugs or have feature requests, please [open an issue](https://github.com/adt-kmr/Ichthys/issues) on the GitHub repository.

## License

This project is licensed under the MIT License.