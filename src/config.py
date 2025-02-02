import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models", "saved_models")

# Model configuration
MODEL_CONFIG = {
    "model_name": "microsoft/resnet-50",
    "image_size": (140, 140),
    "batch_size": 16,
    "num_epochs": 10,
    "learning_rate": 1e-4,
    "num_classes": 2,
}

# Training configuration
TRAIN_CONFIG = {
    "train_split": 0.7,
    "val_split": 0.15,
    "test_split": 0.15,
    "random_seed": 42,
}

# Data augmentation configuration
AUGMENTATION_CONFIG = {
    "rotation_range": 15,
    "width_shift_range": 0.1,
    "height_shift_range": 0.1,
    "horizontal_flip": True,
    "fill_mode": "nearest"
}

# Flask application configuration
FLASK_CONFIG = {
    "SECRET_KEY": "your-secret-key-here",
    "MAX_CONTENT_LENGTH": 16 * 1024 * 1024,  # 16MB max file size
    "UPLOAD_FOLDER": os.path.join(BASE_DIR, "webapp", "static", "uploads"),
    "ALLOWED_EXTENSIONS": {"png", "jpg", "jpeg"}
}