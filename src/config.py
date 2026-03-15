import os
from pathlib import Path

# 1. PATH RESOLUTION
# This dynamically finds the root folder of your project, regardless of 
# whether you are running this on Windows, Mac, or Linux.
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
MODELS_DIR = BASE_DIR / "models"

# Automatically create these directories if they don't exist yet
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# 2. MACHINE LEARNING HYPERPARAMETERS
# EfficientNetB3 (which we will use) optimally takes 300x300, but 224x224 
# is the standard for most CNNs and uses less memory. Let's stick to 224.
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)

BATCH_SIZE = 32
EPOCHS = 1  # We can adjust this later if training takes too long
NUM_CLASSES = 5

# 3. CLASS MAPPINGS
# These match the exact subfolders in the Kaggle dataset
CLASS_NAMES = ["Healthy", "Mild DR", "Moderate DR", "Proliferate DR", "Severe DR"]

# Kaggle Dataset Identifier
KAGGLE_DATASET = "sachinkumar413/diabetic-retinopathy-dataset"