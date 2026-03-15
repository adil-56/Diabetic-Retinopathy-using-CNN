import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras import layers, models
from src.config import IMG_SIZE, NUM_CLASSES

def build_model(learning_rate=1e-3):
    """
    Builds and compiles the EfficientNetB3 transfer learning model.
    """
    print("Building EfficientNetB3 model architecture...")

    # 1. Load the Base Model
    # include_top=False means we do not load the original 1000-class classifier
    # weights='imagenet' loads the pre-trained weights
    base_model = EfficientNetB3(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        include_top=False,
        weights='imagenet'
    )

    # Freeze the base model layers so they don't get updated during initial training
    base_model.trainable = False 

    # 2. Build the Custom Classifier Head
    model = models.Sequential([
        base_model,
        
        # Shrink the 2D feature maps down to a 1D vector
        layers.GlobalAveragePooling2D(),
        
        # Add a dense layer to learn complex combinations of features
        layers.Dense(256, activation='relu'),
        
        # Normalize the activations to speed up training and add stability
        layers.BatchNormalization(),
        
        # Drop 50% of the neurons randomly to prevent the model from memorizing the data
        layers.Dropout(0.5),
        
        # Final classification layer (5 neurons for 5 classes, softmax for probabilities)
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    # 3. Compile the Model
    # CategoricalCrossentropy is standard for multi-class classification
    # We include AUC (Area Under the Curve) as a metric because it is standard in healthcare
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    return model

# Quick test block to verify the architecture without training
if __name__ == "__main__":
    test_model = build_model()
    test_model.summary()