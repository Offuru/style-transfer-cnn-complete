import copy
import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from config import Config
import image_utils
import mlflow_utils
import models
import argparse
from pathlib import Path

device = None
image_size = None
image_transform = None
vgg = None


def setup():
    global device, image_size, image_transform, vgg

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = Config.IMG_SIZE_GPU if device.type == "cuda" else Config.IMG_SIZE_CPU
    image_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )
    vgg = models.get_vgg19_model().features.to(device)
    
    img_path = Path("img")
    
    if not (img_path / "content").exists():
        (img_path / "content").mkdir(parents=True, exist_ok=True)
    if not (img_path / "style").exists():
        (img_path / "style").mkdir(parents=True, exist_ok=True)
    if not (img_path / "generated").exists():
        (img_path / "generated").mkdir(parents=True, exist_ok=True)


def get_style_transfer_model_and_features(
    cnn,
    normalization_mean,
    normalization_std,
    style_img,
    content_img,
    content_layers=None,
    style_layers=None,
):
    if content_layers is None:
        content_layers = Config.CONTENT_LAYERS
    if style_layers is None:
        style_layers = Config.STYLE_LAYERS
        
        
    cnn = copy.deepcopy(cnn)
    
    normalization_mean = torch.tensor(normalization_mean).view(-1, 1, 1).to(device)
    normalization_std = torch.tensor(normalization_std).view(-1, 1, 1).to(device)
    
    normalization = models.NormalizationLayer(normalization_mean, normalization_std).to(device)
    
    
    content_losses = []
    style_losses = []
    
    style_transfer_model = nn.Sequential(normalization)
        
    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f"conv_{i}"
        elif isinstance(layer, nn.ReLU):
            name = f"relu_{i}"
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f"pool_{i}"
        elif isinstance(layer, nn.BatchNorm2d):
            name = f"bn_{i}"
        else:
            raise RuntimeError(f"Unrecognized layer type: {layer}")
        
        style_transfer_model.add_module(name, layer)
        
        if name in content_layers:
            target = style_transfer_model(content_img).detach()
            content_loss = models.ContentLoss(target).to(device)
            style_transfer_model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)
        
        if name in style_layers:
            target_feature = style_transfer_model(style_img).detach()
            style_loss = models.StyleLoss(target_feature).to(device)
            style_transfer_model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)

    for i in range(len(style_transfer_model) - 1, -1, -1):
        if isinstance(style_transfer_model[i], (models.ContentLoss, models.StyleLoss)):
            break
        
    style_transfer_model = style_transfer_model[:i + 1]
    
    return style_transfer_model, content_losses, style_losses


def run_style_transfer(
    cnn, 
    normalization_mean,
    normalization_std,
    content_img,
    style_img,
    input_img=None,
    num_steps=None,
    style_weight=None,
    content_weight=None
):
    num_steps = num_steps or Config.NUM_STEPS
    style_weight = style_weight or Config.STYLE_WEIGHT
    content_weight = content_weight or Config.CONTENT_WEIGHT
    if input_img is None:
        input_img = torch.randn_like(content_img).to(device)    
    
    
    params = {
        "num_steps": num_steps,
        "style_weight": style_weight,
        "content_weight": content_weight,
        "image_size": image_size,
        "content_layers": Config.CONTENT_LAYERS,
        "style_layers": Config.STYLE_LAYERS,
        "normalization_mean": normalization_mean,
        "normalization_std": normalization_std,
    }
    mlflow.log_params(params)
    
    style_transfer_model, content_losses, style_losses = get_style_transfer_model_and_features(
        cnn, normalization_mean, normalization_std, style_img, content_img
    )
    
    input_img.requires_grad_(True)
    optimizer = optim.LBFGS([input_img])
    
    run = [0]
    
    while run[0] < num_steps:
        
        def closure():
            with torch.no_grad():
                input_img.clamp_(0, 1)
            
            optimizer.zero_grad()
            model_output = style_transfer_model(input_img)
            
            content_loss = sum(content_loss.loss for content_loss in content_losses)
            style_loss = sum(style_loss.loss for style_loss in style_losses)
            
            total_loss = content_weight * content_loss + style_weight * style_loss
            
            total_loss.backward()
            
            run[0] += 1
            
            if run[0] % 50 == 0:
                print(f"Run {run[0]}: Total loss - {total_loss.item()}, Content loss - {content_loss.item()}, Style loss - {style_loss.item()}")
                
                mlflow.log_metric("total_loss", total_loss.item(), step=run[0])
                mlflow.log_metric("content_loss", content_loss.item(), step=run[0])
                mlflow.log_metric("style_loss", style_loss.item(), step=run[0])
                
            return total_loss
        
        optimizer.step(closure)
        
    with torch.no_grad():
        input_img.clamp_(0, 1)

    # ========= 
    # mlflow_utils.log_model_info(style_transfer_model)
    # mlflow_utils.log_trained_model(style_transfer_model, image_size, device)
    
    # Used for saving the model weights and architecture, but it doesn't really work in this case,
    # because it only works if we have a model that generates things or computes something for example,
    # but in this case, style_transfer_model is just a feature extractor used for computing losses during training

    # after the training, you can load the model by calling mlflow_utils.load_model(run_id), (you can find run_id in the MLflow UI)
    # ========= 

    return input_img

def main(args):
    setup()
    
    mlflow.set_tracking_uri(uri=Config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(Config.MLFLOW_EXPERIMENT_NAME)
    
    with mlflow.start_run():
        
        content_path = f"img/content/{args.content}"
        style_path = f"img/style/{args.style}"

        content_img = image_utils.load_image(content_path, image_transform, device)
        style_img = image_utils.load_image(style_path, image_transform, device)
                
        input_img = content_img.clone().to(device) if args.use_content_as_input else None
        
        normalization_mean = Config.MEAN
        normalization_std = Config.STD
        
        output = run_style_transfer(
            vgg,
            normalization_mean,
            normalization_std,
            content_img,
            style_img,
            input_img=input_img,
            num_steps=Config.NUM_STEPS,
            style_weight=Config.STYLE_WEIGHT,
            content_weight=Config.CONTENT_WEIGHT
        )
        
        generated_image = image_utils.tensor_to_image(output)
        generated_path = f"img/generated/{args.generated}"
        generated_image.save(generated_path)
        
        mlflow.log_artifact(generated_path)
        mlflow.log_artifact(content_path)
        mlflow.log_artifact(style_path)
        
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Style Transfer using VGG19")
    
    parser.add_argument("--content", type=str, help="Name of the content image, from img/content/")   
    parser.add_argument("--style", type=str, help="Name of the style image, from img/style/")
    parser.add_argument("--generated", type=str, default="output.png", help="Name of the generated image to save")
    parser.add_argument("--use_content_as_input", type=bool, default=True, help="Use content image as input for style transfer, default is True, set to False to use random noise") 
    
    args = parser.parse_args()
    
    main(args)