class Config:
    # Image size
    IMG_SIZE_GPU = 512
    IMG_SIZE_CPU = 256

    # Hyperparameters
    NUM_STEPS = 200
    LOG_INTERVAL = 50
    STYLE_WEIGHT = 1e6
    CONTENT_WEIGHT = 1
    CONTENT_LAYERS = ["conv_4"]
    STYLE_LAYERS = ["conv_1", "conv_2", "conv_3", "conv_4", "conv_5"]

    # ImageNet1K normalization tensors
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    # Mlflow parameters
    MLFLOW_EXPERIMENT_NAME = "style_transfer"
    MLFLOW_TRACKING_URI = "http://localhost:8888"
