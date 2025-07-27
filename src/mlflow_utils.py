import mlflow
import torch
from mlflow.models import infer_signature
import mlflow.pytorch


def log_model_info(model):
    
    model_summary = str(model)
    mlflow.log_text(model_summary, "model_summary.txt")
    
    layer_info = []
    for name, module in model.named_modules():
        layer_info.append(f"{name}: {module.__class__.__name__}")
    mlflow.log_text("\n".join(layer_info), "layer_info.txt")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    mlflow.log_param("total_params", total_params)
    mlflow.log_param("trainable_params", trainable_params)
    
    
@torch.no_grad()
def log_trained_model(model, imsize, device):
    input_example = torch.randn(1, 3, imsize, imsize).to(device)
    output_example = model(input_example)
    
    import logging
    logging.getLogger("mlflow").setLevel(logging.ERROR)

    signature = infer_signature(
        input_example.cpu().numpy(),
        output_example.detach().cpu().numpy()
    )
    
    mlflow.pytorch.log_model(
        model, "style_transfer_model", 
        input_example=input_example.cpu().numpy(),
        signature=signature
    )


def load_model(run_id):
    """
    Load a model from a specific run ID.
    """
    model_uri = f"runs:/{run_id}/style_transfer_model"
    return mlflow.pytorch.load_model(model_uri)
    
    