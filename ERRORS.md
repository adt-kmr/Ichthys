# Ichthys Error Messages Guide

This document explains the different error categories you might encounter when using Ichthys, along with suggested fixes.

## File Not Found Errors

### `checkpoint not found` (infer.py)
**Cause**: The checkpoint `.pth` file specified does not exist at the given path.

**Suggested fixes**:
- Verify the checkpoint path is correct
- Ensure the `.pth` file exists in the specified location
- Use `ichthys-viz` to browse available checkpoints in your workspace
- List checkpoints: `python -m viz.jobs list` (if available)

### `prediction file not found` (evaluate.py)
**Cause**: The prediction text file specified with `--pred` does not exist.

**Suggested fixes**:
- Check the file path is correct
- Ensure the file has the expected header: `object,Timestamp,X,Y,Z,group`
- Generate a prediction file using: `python infer.py <ckpt> <scene_json> <images_root> --no_dino --output <output>`

### `ground truth file not found` (evaluate.py)
**Cause**: The ground truth CSV file specified with `--gt` does not exist.

**Suggested fixes**:
- Verify the GT file path is correct
- Ensure the file has the expected columns: `Actor,Timestamp,X,Y,Z`
- Generate GT data from your tracking results or use the provided SynFish datasets

### `workspace root does not exist` (app.py/dashboard)
**Cause**: The workspace root directory specified does not exist.

**Suggested fixes**:
- Use `streamlit run app.py -- --workspace /path/to/valid/outputs`
- Or upload prediction/GT files via the sidebar to use temporary storage
- Ensure the path is absolute or relative to your current working directory

## Argument Errors

### `missing required argument: train_folder` (train.py)
**Cause**: The required `train_folder` argument was not provided.

**Suggested fixes**:
- Add the training folder as the first argument: `python train.py /path/to/training/data`
- See: `python train.py --help` for all available options

### `invalid choice` (train.py)
**Cause**: An invalid choice was provided for `--selection_metric` or other choice arguments.

**Suggested fixes**:
- Valid choices for `--selection_metric`: `train` or `val`
- Valid choices for other arguments are shown in `--help` output

## Runtime Errors

### `CUDA out of memory`
**Cause**: Running inference/training with insufficient GPU memory.

**Suggested fixes**:
- Use `--fp16` flag for mixed precision
- Reduce batch size or number of workers
- Use a GPU with more memory

### `Checkpoint has no predictor_state_dict`
**Cause**: The checkpoint was trained without the temporal prediction head.

**Suggested fixes**:
- Use `--no_predictor` flag to run in ablation mode
- Or use a checkpoint that was trained with `--use_dino` or full training

### `Unknown benchmark GT filename`
**Cause**: The GT filename doesn't match the expected pattern for automatic gate selection.

**Suggested fixes**:
- Pass `--dist_thr` explicitly to set a custom matching threshold
- Rename your GT file to match the pattern: `test1-gt.csv`, `test2-gt.csv`, etc.
- Or use a custom threshold: `python evaluate.py --pred ... --gt ... --dist_thr 0.75`

## Completion System Errors

### `bash: ichthys: command not found`
**Cause**: The Ichthys CLI is not installed or not in PATH.

**Suggested fixes**:
- Install the package: `pip install -e .`
- Or use: `python -m train`, `python -m infer`, etc.
- Or add completion script to your shell: `source scripts/ichthys-completion.sh`

### `no completions available`
**Cause**: The bash/zsh completion script is not sourced or properly installed.

**Suggested fixes**:
- Bash: `source scripts/ichthys-completion.sh`
- Zsh: Add to `.zshrc`: `fpath+=(~/scripts)` and `autoload -U compinit && compinit`
- Re-run your shell after sourcing the completion script

## Contributors

- @adt-kmr - initial implementation