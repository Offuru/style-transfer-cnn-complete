@echo off
echo Starting MLflow server...

poetry run mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --port 8888

pause
