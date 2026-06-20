# ScanNet++ Official Preprocessing

This directory vendors the ScanNet++ official preprocessing scripts that are needed to generate project inputs from the raw ScanNet++ release.

## Files

- `scannetpp_convert_sg.py`
- `merge_sg.py`
- `my_config.yml`
- `requirements.txt`

## Environment

Use a separate Python environment for these scripts. They depend on the official ScanNet++ toolchain and an older PyTorch stack.

Recommended order:

1. Set up the main `CRISP_bench` environment first.
2. Set up the `lmms-eval` environment if you need model evaluation.
3. Only then create this ScanNet++-specific environment if you need to generate ScanNet++ data yourself.

## Scope

This directory is only for ScanNet++ data generation. It is not required for users who only want to run evaluation on already prepared data.
