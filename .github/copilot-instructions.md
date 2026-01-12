# Copilot Instructions for BBDM Project

This project implements Brownian Bridge Diffusion Models (BBDM) using PyTorch. The architecture is configuration-driven and relies heavily on a registry pattern for dynamic component instantiation.

## 🏗 Project Architecture

- **Entry Point**: `main.py` is the universal entry point. It handles CLI parsing (argparse), configuration loading (YAML), and process launching (DDP or single-device).
- **Configuration**:
  - Located in `configs/*.yaml`.
  - Loaded into nested `argparse.Namespace` objects (via `utils.dict2namespace`), allowing dot-notation access (e.g., `config.model.EMA.use_ema`).
  - **Crucial**: Always modify options in the YAML config, not hardcoded in python files.
- **Registry Pattern**:
  - `Register.py` is the central registry (imported as `Registers`).
  - Components (Runners, Models, Datasets) must be decorated to be discoverable:
    ```python
    @Registers.runners.register_with_name('MyRunner')
    class MyRunner(BaseRunner): ...
    ```
- **Runners**:
  - Training/Testing logic resides in `runners/`.
  - Hierarchy: `BaseRunner` -> `DiffusionBaseRunner` -> `BBDMRunner`.
  - `BBDMRunner.py` contains the specific logic for Brownian Bridge Diffusion.

## 🚀 Key Workflows

### 1. Training & Testing
Control execution via `main.py` flags.
- **Train**:
  ```bash
  python main.py -c configs/Template-BBDM.yaml -t --gpu_ids 0
  ```
- **Test/Sample** (omit `-t`):
  ```bash
  python main.py -c configs/Template-BBDM.yaml --gpu_ids 0 --resume_model results/checkpoints/last.pth
  ```
- **Distributed Training (DDP)**:
  Pass multiple IDs to `gpu_ids`. The script handles `torch.multiprocessing`.
  ```bash
  python main.py -c configs/Template-BBDM.yaml -t --gpu_ids 0,1,2,3
  ```

### 2. Adding a New Dataset
1.  Create/Modify a file in `datasets/` (e.g., `datasets/custom.py`).
2.  Inherit from `torch.utils.data.Dataset`.
3.  **Register the class**:
    ```python
    from Register import Registers
    @Registers.datasets.register_with_name('my_new_dataset')
    class MyDataset(Dataset): ...
    ```
4.  Update the YAML config `dataset_name` to match the registered name (`my_new_dataset`).

### 3. Adding a New Model
1.  Define model in `model/`.
2.  Register via `@Registers.models.register_with_name('ModelName')`.
3.  Update YAML `model.model_type` to `'ModelName'`.

## 🛠 Project Conventions

- **Paths**: Dataset paths in YAML are absolute or relative to the project root. Structure expectations (e.g., `train/A`, `train/B`) are defined in the Dataset class logic.
- **Random Seed**: Handled centrally via `utils.set_random_seed` called in `main.py`.
- **Dependencies**: Managed via Conda (`environment.yml`).
- **Code Style**:
  - Prefer `argparse.Namespace` over dictionaries for passing config context.
  - Use `os.path.join` for cross-platform compatibility, though `pathlib` is present in some newer files.

## ⚠️ Common Pitfalls
- **Runner Resolution**: `utils.get_runner` uses the string name from YAML (`runner: "BBDMRunner"`) to look up the class in `Registers`. Ensure names match exactly.
- **Variable Scope**: `config` object is passed down through most constructors. Avoid shadowing it.
