import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
import os

# --- Configuration ---
IMG_WIDTH, IMG_HEIGHT = 224, 224
BATCH_SIZE = 32
EPOCHS = 10 # Start with 10, increase if you have time and see improvement
DATA_DIR = 'data'

# --- 1. Load Data ---
print("Loading and preprocessing data...")
# Create a dataset from the image directories
train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2, # Use 20% of the images for validation
    subset="training",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
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
# Load the pre-trained MobileNetV2 model, without its top classification layer
base_model = MobileNetV2(input_shape=(IMG_WIDTH, IMG_HEIGHT, 3),
                         include_top=False,
                         weights='imagenet')

# Freeze the base model layers
base_model.trainable = False

# Create a new model on top
model = models.Sequential([
    # Add a rescaling layer to normalize pixel values
    layers.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5), # Add dropout for regularization
    layers.Dense(len(class_names), activation='softmax') # Output layer
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
model.save('cattleholic_model.h5')
print("Model saved as cattleholic_model.h5! You are ready for the next phase.")