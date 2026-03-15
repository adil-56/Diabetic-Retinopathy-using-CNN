import os
import shutil
import kagglehub
import tensorflow as tf
from src.config import RAW_DATA_DIR, KAGGLE_DATASET, BATCH_SIZE, IMG_SIZE

def download_and_prepare_data():
    """
    Downloads the dataset from Kaggle and moves it to the local data/raw directory.
    Skips the download if the data is already present.
    """
    # Check if the directory is empty
    if not os.path.exists(RAW_DATA_DIR) or not os.listdir(RAW_DATA_DIR):
        print(f"Downloading dataset '{KAGGLE_DATASET}' from Kaggle...")
        
        # kagglehub downloads to a hidden cache folder by default
        cache_path = kagglehub.dataset_download(KAGGLE_DATASET)
        print(f"Dataset downloaded to cache: {cache_path}")
        
        print("Moving data to local project folder (data/raw/)...")
        # Move contents from the cache to our structured project folder
        for item in os.listdir(cache_path):
            source = os.path.join(cache_path, item)
            destination = os.path.join(RAW_DATA_DIR, item)
            if os.path.isdir(source):
                shutil.copytree(source, destination, dirs_exist_ok=True)
            else:
                shutil.copy2(source, destination)
                
        print("Data preparation complete!")
    else:
        print("Data already exists locally. Skipping download.")

def create_data_loaders():
    """
    Creates high-performance tf.data.Dataset objects for training and validation.
    """
    print("Loading data into TensorFlow datasets...")
    
    # Create the Training Dataset (80% of the data)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        RAW_DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=123,  # Seed ensures no overlap between train and validation
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical' # 'categorical' because we have 5 classes
    )

    # Create the Validation Dataset (20% of the data)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        RAW_DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical'
    )

    # Performance Optimization: Prefetching and Caching
    AUTOTUNE = tf.data.AUTOTUNE
    
    # Cache keeps the images in memory after the first epoch
    # Shuffle mixes the data so the model doesn't memorize the order
    # Prefetch prepares the next batch while the current batch is training
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds

# Quick test block: If you run this file directly, it will just download the data.
if __name__ == "__main__":
    download_and_prepare_data()