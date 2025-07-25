# Diabetic Retinopathy Detection using CNN
# Complete implementation with data preprocessing, model training, and evaluation

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import cv2
import os
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class DiabeticRetinopathyDetector:
    def __init__(self, img_size=(224, 224), num_classes=5):
        self.img_size = img_size
        self.num_classes = num_classes
        self.model = None
        self.history = None
        
    def preprocess_images(self, image_paths, labels=None):
        """
        Preprocess retinal images for training/prediction
        """
        images = []
        processed_labels = []
        
        for i, img_path in enumerate(image_paths):
            try:
                # Load and resize image
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, self.img_size)
                
                # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
                img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
                img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
                
                # Normalize pixel values
                img = img.astype('float32') / 255.0
                
                images.append(img)
                if labels is not None:
                    processed_labels.append(labels[i])
                    
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
                continue
                
        return np.array(images), np.array(processed_labels) if labels is not None else None
    
    def create_advanced_cnn_model(self):
        """
        Create an advanced CNN model for diabetic retinopathy detection
        """
        model = keras.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*self.img_size, 3)),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth Convolutional Block
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Global Average Pooling
            layers.GlobalAveragePooling2D(),
            
            # Dense layers
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def create_transfer_learning_model(self):
        """
        Create a transfer learning model using EfficientNetB0
        """
        base_model = keras.applications.EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.img_size, 3)
        )
        
        # Freeze base model initially
        base_model.trainable = False
        
        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model, base_model
    
    def compile_model(self, model, learning_rate=0.001):
        """
        Compile the model with appropriate optimizer and loss function
        """
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        return model
    
    def create_callbacks(self, model_name='diabetic_retinopathy_model'):
        """
        Create callbacks for training
        """
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-7
            ),
            keras.callbacks.ModelCheckpoint(
                f'{model_name}.h5',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max'
            )
        ]
        return callbacks
    
    def train_model(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32, use_transfer_learning=True):
        """
        Train the diabetic retinopathy detection model
        """
        if use_transfer_learning:
            self.model, base_model = self.create_transfer_learning_model()
        else:
            self.model = self.create_advanced_cnn_model()
            base_model = None
        
        self.model = self.compile_model(self.model)
        
        # Print model summary
        print("Model Architecture:")
        self.model.summary()
        
        # Create callbacks
        callbacks = self.create_callbacks()
        
        # Train the model
        print("Starting training...")
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # Fine-tuning for transfer learning
        if use_transfer_learning and base_model is not None:
            print("Starting fine-tuning...")
            base_model.trainable = True
            
            # Use a lower learning rate for fine-tuning
            self.model = self.compile_model(self.model, learning_rate=0.0001)
            
            # Continue training with fine-tuning
            fine_tune_epochs = 20
            total_epochs = len(self.history.history['loss']) + fine_tune_epochs
            
            history_fine = self.model.fit(
                X_train, y_train,
                batch_size=batch_size,
                epochs=total_epochs,
                initial_epoch=len(self.history.history['loss']),
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=1
            )
            
            # Combine histories
            for key in self.history.history.keys():
                self.history.history[key].extend(history_fine.history[key])
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the trained model
        """
        if self.model is None:
            print("Model not trained yet!")
            return
        
        # Make predictions
        y_pred_proba = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Calculate metrics
        test_loss, test_acc, test_precision, test_recall = self.model.evaluate(X_test, y_test, verbose=0)
        
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test Precision: {test_precision:.4f}")
        print(f"Test Recall: {test_recall:.4f}")
        
        # Classification report
        class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=class_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix - Diabetic Retinopathy Detection')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
        
        return y_pred, y_pred_proba
    
    def plot_training_history(self):
        """
        Plot training history
        """
        if self.history is None:
            print("No training history available!")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Training Loss')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision
        axes[1, 0].plot(self.history.history['precision'], label='Training Precision')
        axes[1, 0].plot(self.history.history['val_precision'], label='Validation Precision')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Recall
        axes[1, 1].plot(self.history.history['recall'], label='Training Recall')
        axes[1, 1].plot(self.history.history['val_recall'], label='Validation Recall')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def predict_single_image(self, image_path):
        """
        Predict diabetic retinopathy for a single image
        """
        if self.model is None:
            print("Model not trained yet!")
            return
        
        # Preprocess image
        X, _ = self.preprocess_images([image_path])
        
        # Make prediction
        prediction_proba = self.model.predict(X)
        prediction = np.argmax(prediction_proba, axis=1)[0]
        confidence = np.max(prediction_proba)
        
        class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
        
        print(f"Prediction: {class_names[prediction]}")
        print(f"Confidence: {confidence:.4f}")
        
        # Display image with prediction
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(8, 6))
        plt.imshow(img)
        plt.title(f'Prediction: {class_names[prediction]} (Confidence: {confidence:.2f})')
        plt.axis('off')
        plt.show()
        
        return prediction, confidence

# Example usage and data simulation
def simulate_training_data(num_samples=1000):
    """
    Simulate training data structure (replace with actual data loading)
    """
    print("Note: This is a simulation. Replace with actual retinal image dataset.")
    print("Popular datasets: APTOS 2019, Kaggle Diabetic Retinopathy Detection")
    
    # Simulate image paths and labels
    image_paths = [f"retinal_image_{i}.jpg" for i in range(num_samples)]
    labels = np.random.randint(0, 5, num_samples)  # 5 classes: 0-4
    
    return image_paths, labels

def main():
    """
    Main training pipeline
    """
    # Initialize detector
    detector = DiabeticRetinopathyDetector()
    
    # Simulate data (replace with actual data loading)
    print("Loading dataset...")
    image_paths, labels = simulate_training_data(1000)
    
    # Note: In real implementation, you would:
    # 1. Download dataset (APTOS 2019, Kaggle DR dataset)
    # 2. Load actual images using detector.preprocess_images()
    # 3. Split into train/validation/test sets
    
    print("Dataset loaded successfully!")
    print(f"Total samples: {len(image_paths)}")
    print(f"Class distribution: {np.bincount(labels)}")
    
    # For demonstration, create dummy data
    X_dummy = np.random.rand(100, 224, 224, 3).astype('float32')
    y_dummy = np.random.randint(0, 5, 100)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_dummy, y_dummy, test_size=0.2, random_state=42, stratify=y_dummy
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    # Train model (uncomment for actual training)
    # detector.train_model(X_train, y_train, X_val, y_val, epochs=50)
    
    # Plot training history
    # detector.plot_training_history()
    
    # Evaluate model
    # detector.evaluate_model(X_test, y_test)
    
    print("\nProject setup complete!")
    print("Next steps:")
    print("1. Download diabetic retinopathy dataset (APTOS 2019 or Kaggle)")
    print("2. Replace dummy data with actual image preprocessing")
    print("3. Run training with actual data")
    print("4. Fine-tune hyperparameters")
    print("5. Add data augmentation for better performance")

if __name__ == "__main__":
    main()
