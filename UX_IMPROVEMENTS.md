# Ichthys UX Improvements

This document summarizes the user experience improvements added to the Ichthys project.

## Command Completion

- **Bash completion**: `scripts/ichthys-completion.sh` - Provides tab completion for `ichthys-train`, `ichthys-infer`, `ichthys-evaluate`, `ichthys-sort3d`, and `ichthys-viz` commands
- **Zsh completion**: `scripts/ichthys-completion.zsh` - Provides tab completion with command descriptions
- **Installation**: Add the completion scripts to your shell profile or use the pyproject.toml configuration

## Error Messages

- **File not found errors**: Clear error messages with suggested fixes for missing checkpoints, predictions, ground truth, and workspace roots
- **Error categories**: Documented in `ERRORS.md` with categories for file not found, argument errors, runtime errors, and progress indicators
- **Suggested fixes**: Each error message includes specific actions the user can take

## Progress Indicators

- **Training engine**: `tqdm` progress bar with `mininterval=5` and `miniters=1` for consistent behavior
- **Inference**: Progress bars for encoding frames and tracking frames with descriptive labels
- **Evaluation**: Progress bar for evaluating frames with descriptive label
- **Dashboard**: Sidebar stats showing predictions, ground truth, and scenes count
- **Format standardization**: All progress bars use the same format: `desc="<action> <object>"` with optional `mininterval` and `miniters` parameters

## Error Message Categories

- **File Not Found**: Checkpoint, prediction, GT, workspace root
- **Argument Errors**: Missing required args, invalid choices
- **Runtime Errors**: CUDA OOM, checkpoint without predictor, unknown GT filename
- **Progress Indicators**: Tqdm bar format and behavior

## Keyboard Shortcuts (Dashboard)

- **`r`** - Rescan workspace
- **`p`** - Focus prediction file selector
- **`g`** - Focus ground truth file selector
- **`Tab`** - Navigate between sidebar elements

Documented in the dashboard sidebar under "Keyboard Shortcuts" expander.

## Progress Indicators

Progress bars use `tqdm` and show:
- Current progress percentage and bar
- Elapsed time since start
- Estimated time remaining
- Descriptive label (e.g., "Training epochs", "Encoding frames")

Progress bars automatically update per iteration and disappear when complete.

## Version
- Added: 2026-01-01
- Initial UX improvements release