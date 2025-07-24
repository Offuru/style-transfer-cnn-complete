from PIL import Image
from torchvision import transforms


def load_image(image_path, image_transform, device):
    try:
        image = Image.open(image_path)
        image = image_transform(image).unsqueeze(0).to(device)
        return image
    except FileNotFoundError:
        print(f"Error: The file {image_path} was not found.")
    except Exception as e:
        print(f"An error occurred while loading the image: {e}")


def tensor_to_image(tensor):
    image = tensor.cpu().clone().squeeze(0)
    image = transforms.ToPILImage()(image)
    return image


def validate_image_sizes(image1, image2):
    if image1.size != image2.size:
        raise ValueError(f"Image sizes do not match: {image1.size} vs {image2.size}")
    return True
