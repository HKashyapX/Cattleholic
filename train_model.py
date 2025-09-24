import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
import os
from pathlib import Path

# --- Configuration ---
IMG_WIDTH, IMG_HEIGHT = 224, 224
BATCH_SIZE = 32
EPOCHS = 600
# Use Path for proper path handling across platforms
DATA_DIR = Path(os.path.abspath('data'))

# --- 1. Load Data ---
print("Loading and preprocessing data...")

try:
    # Verify directory exists
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")
        
    # Create a dataset from the image directories
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        str(DATA_DIR),  # Convert Path to string
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        str(DATA_DIR),  # Convert Path to string
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )

    class_names = train_dataset.class_names
    print(f"Found classes: {class_names}")

    # --- 2. Build the Model (Transfer Learning) ---
    print("Building the model...")
    base_model = MobileNetV2(input_shape=(IMG_WIDTH, IMG_HEIGHT, 3),
                            include_top=False,
                            weights='imagenet')

    base_model.trainable = False

    model = models.Sequential([
        layers.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(len(class_names), activation='softmax')
    ])

    # --- 3. Compile the Model ---
    print("Compiling the model...")
    model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])

    model.summary()

    # --- 4. Train the Model ---
    print(f"Starting training for {EPOCHS} epochs...")
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS
    )

    # --- 5. Save the Model ---
    print("Training complete. Saving model...")
    save_path = Path('cattleholic_model.h5')
    model.save(str(save_path))
    print(f"Model saved as {save_path}!")

except Exception as e:
    print(f"An error occurred: {str(e)}")
    raise