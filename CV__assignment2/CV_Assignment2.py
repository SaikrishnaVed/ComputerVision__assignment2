import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, MultiHeadAttention, Reshape, GlobalAveragePooling1D, Dropout, BatchNormalization # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore

# 1. Data Preprocessing & Augmentation

def load_and_preprocess_image(image_path, target_size=(128, 128)):
    """Load and preprocess an image."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Unable to load image at path: {image_path}")
    
    image = cv2.resize(image, target_size)
    image = image.astype('float32') / 255.0  # Normalize to [0, 1]
    return image

# Example paths (replace with actual paths)
path1 = "Assets/ct1000.png"
path2 = "Assets/mri1000.jpg"

# Load and preprocess images
ct_image = load_and_preprocess_image(path1)
mri_image = load_and_preprocess_image(path2)

# Add channel dimension for model input
ct_image = np.expand_dims(ct_image, axis=-1)
mri_image = np.expand_dims(mri_image, axis=-1)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True
)

# Apply augmentation (example for a single image)
ct_image_augmented = datagen.random_transform(ct_image)
mri_image_augmented = datagen.random_transform(mri_image)

# 2. Cross-Modal Feature Extraction & Fusion

def create_feature_extractor(input_shape):
    """Create a feature extractor using a CNN."""
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    return tf.keras.Model(inputs, x)

def cross_modal_attention(ct_features, mri_features):
    """Apply cross-modal attention."""
    # Reshape features for MultiHeadAttention
    ct_features = Reshape((-1, 64))(ct_features)  # Adjust shape based on feature dimensions
    mri_features = Reshape((-1, 64))(mri_features)
    
    # Apply attention
    attention_output = MultiHeadAttention(num_heads=4, key_dim=32)(ct_features, mri_features)  # Reduced num_heads and key_dim
    return attention_output

# Input layers
ct_input = Input(shape=(128, 128, 1))  # Reduced input size
mri_input = Input(shape=(128, 128, 1))  # Reduced input size

# Feature extractors
ct_feature_extractor = create_feature_extractor((128, 128, 1))  # Reduced input size
mri_feature_extractor = create_feature_extractor((128, 128, 1))  # Reduced input size

# Extract features
ct_features = ct_feature_extractor(ct_input)
mri_features = mri_feature_extractor(mri_input)

# Fuse features using cross-modal attention
fused_features = cross_modal_attention(ct_features, mri_features)

# Global average pooling to reduce output shape to (None, 1)
x = GlobalAveragePooling1D()(fused_features)

# Final dense layers
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)  # Add dropout for regularization
outputs = Dense(1, activation='sigmoid')(x)  # Binary classification

# Build the model
model = tf.keras.Model(inputs=[ct_input, mri_input], outputs=outputs)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# 3. Training the Model

# Example training data (replace with actual data)
train_ct_images = np.random.rand(100, 128, 128, 1)  # Reduced input size
train_mri_images = np.random.rand(100, 128, 128, 1)  # Reduced input size
train_labels = np.random.randint(0, 2, size=(100,))  # Example: Binary labels for 100 samples

# Callbacks for better training
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

# Train the model
history = model.fit(
    [train_ct_images, train_mri_images],
    train_labels,
    epochs=50,  # Increased epochs
    batch_size=16,  # Reduced batch size
    validation_split=0.2,
    callbacks=[early_stopping, reduce_lr]
)

# 4. Evaluation & Benchmarking

# Example test data (replace with actual data)
test_ct_images = np.random.rand(10, 128, 128, 1)  # Reduced input size
test_mri_images = np.random.rand(10, 128, 128, 1)  # Reduced input size
test_labels = np.random.randint(0, 2, size=(10,))  # Example: Binary labels for 10 samples

# Evaluate model
def evaluate_model(model, test_ct, test_mri, test_labels):
    # Get predictions (probabilities)
    predictions = model.predict([test_ct, test_mri])
    
    # Convert probabilities to binary predictions (0 or 1)
    binary_predictions = (predictions > 0.5).astype(int).flatten()
    
    # Flatten the labels to 1D array
    test_labels = test_labels.flatten()
    
    # Check if there are any positive predictions
    if np.sum(binary_predictions) == 0:
        print("Warning: No positive predictions!")
        precision = 0.0
        recall = 0.0
        f1 = 0.0
    else:
        # Calculate metrics
        precision = precision_score(test_labels, binary_predictions)
        recall = recall_score(test_labels, binary_predictions)
        f1 = f1_score(test_labels, binary_predictions)
    
    # AUC-ROC (requires probabilities)
    auc = roc_auc_score(test_labels, predictions)
    
    print(f"Precision: {precision}, Recall: {recall}, F1-Score: {f1}, AUC-ROC: {auc}")

# Evaluate the model
evaluate_model(model, test_ct_images, test_mri_images, test_labels)

# Plot training and validation loss
import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()