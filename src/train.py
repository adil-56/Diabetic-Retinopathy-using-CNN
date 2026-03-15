import os
import tensorflow as tf
from src.config import EPOCHS, MODELS_DIR
from src.data_pipeline import create_data_loaders
from src.model import build_model

def train_model():
    print("Starting the training pipeline...")

    # 1. Load the data streams
    train_ds, val_ds = create_data_loaders()

    # 2. Initialize the model architecture
    model = build_model()

    # 3. Define Professional Callbacks
    model_save_path = os.path.join(MODELS_DIR, "dr_model_best.keras")
    
    callbacks = [
        # Saves the model only when validation accuracy improves
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_save_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        # Stops training early if the model stops learning
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=4,
            restore_best_weights=True,
            verbose=1
        ),
        # Reduces learning rate when validation loss plateaus
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # 4. Train the model
    print(f"Beginning training for {EPOCHS} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    print(f"Training complete! Best model saved to: {model_save_path}")
    return history

if __name__ == "__main__":
    train_model()