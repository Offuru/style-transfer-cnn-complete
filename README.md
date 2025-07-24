## Installation

Requires [Python 3.11.9](https://www.python.org/downloads/release/python-3119/) 

### Package manager

[Poetry](https://python-poetry.org/) - a lightweight package designed for managing python environments

```console
pip install poetry
```

You can then generate a local venv for your project using

```console
poetry install --no-root
```

This will automatically install the packages defined in `pyproject.toml`

You can add packages using
```console
poetry add "your-package"
```

This project was set up with using CUDA in mind, if you can't run or don't have CUDA installed then delete
```
[tool.poetry.dependencies]
torch = {source = "pytorch_cuda"}
torchvision = {source = "pytorch_cuda"}
```

in `pyproject.toml`. This will make it so the default version of pytorch is installed (keeping it shouldn't lead to errors but the package will take longer to install for no benefits)

## Running the script

First start the mlflow server by running `start_mlflow.bat`. This will be used for logging model training.

```console
python .\src\main.py --content piata_sfatului.png --style starry_night.png --generated starry_brasov.png
```

