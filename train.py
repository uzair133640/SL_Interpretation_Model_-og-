# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
# from tensorflow.keras.models import Model

# # Constants
# IMG_SIZE = (224, 224)
# BATCH_SIZE = 64

# # Load data with augmentation
# train_datagen = ImageDataGenerator(
#      rescale=1./255,
#     rotation_range=30,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     brightness_range=[0.8, 1.2],
#     fill_mode='nearest',
#     validation_split=0.2   
# )

# train_data = train_datagen.flow_from_directory(
#     '/content/SL_Interpretation_Model_-og-/dataset/train',
#     target_size=IMG_SIZE,
#     batch_size=BATCH_SIZE,
#     class_mode='categorical',
#     subset='training'
# )

# val_data = train_datagen.flow_from_directory(
#     '/content/SL_Interpretation_Model_-og-/dataset/train',
#     target_size=IMG_SIZE,
#     batch_size=BATCH_SIZE,
#     class_mode='categorical',
#     subset='validation'
# )

# # Load test data
# test_datagen = ImageDataGenerator(rescale=1./255)
# test_data = test_datagen.flow_from_directory(
#     '/content/SL_Interpretation_Model_-og-/dataset/test',
#     target_size=IMG_SIZE,
#     batch_size=BATCH_SIZE,
#     class_mode='categorical',
#     shuffle=False
# )

# # Build the model
# base_model = MobileNetV2(
#     input_shape=(224, 224, 3),
#     include_top=False,
#     weights='imagenet'
# )
# base_model.trainable = False  # Freeze pre-trained layers

# x = GlobalAveragePooling2D()(base_model.output)
# x = Dense(128, activation='relu')(x)
# output = Dense(len(train_data.class_indices), activation='softmax')(x)

# model = Model(inputs=base_model.input, outputs=output)
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# # Train the model
# history = model.fit(
#     train_data,
#     validation_data=val_data,
#     epochs=50,
#     callbacks=[
#         tf.keras.callbacks.ModelCheckpoint('asl_model.h5', save_best_only=True),
#         tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
#     ]
# )

# # Evaluate on test data
# test_loss, test_acc = model.evaluate(test_data)
# print(f"Test Accuracy: {test_acc*100:.2f}%")


#code for new model

import tensorflow as tf
import numpy as np
import os
from PIL import Image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.utils.class_weight import compute_class_weight

# Constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 32  # Reduced for better convergence
INIT_LR = 3e-4
NUM_EPOCHS = 20

# ----------------------
# Data Validation
# ----------------------
def validate_dataset_structure():
    base_path = '/content/SL_Interpretation_Model_-og-/dataset'
    required_folders = ['train', 'test']
    
    for folder in required_folders:
        path = os.path.join(base_path, folder)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required folder: {path}")
            
        class_folders = os.listdir(path)
        if len(class_folders) != 36:
            raise ValueError(f"Expected 36 class folders in {folder}, found {len(class_folders)}")

def check_image_integrity(folder):
    corrupt_files = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                path = os.path.join(root, file)
                try:
                    with Image.open(path) as img:
                        img.verify()
                except Exception as e:
                    corrupt_files.append(path)
    return corrupt_files

# ----------------------
# Data Pipeline
# ----------------------
def create_data_generators():
    # Verify dataset structure first
    validate_dataset_structure()
    
    # Check for corrupt images
    train_path = '/content/SL_Interpretation_Model_-og-/dataset/train'
    corrupt_train = check_image_integrity(train_path)
    if corrupt_train:
        raise ValueError(f"Found {len(corrupt_train)} corrupt images in training set")

    # Conservative augmentation for ASL
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        fill_mode='nearest',
        validation_split=0.2
    )

    # Training data
    train_data = train_datagen.flow_from_directory(
        '/content/SL_Interpretation_Model_-og-/dataset/train',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    # Validation data
    val_data = train_datagen.flow_from_directory(
        '/content/SL_Interpretation_Model_-og-/dataset/train',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=True
    )

    # Test data
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    test_data = test_datagen.flow_from_directory(
        '/content/SL_Interpretation_Model_-og-/dataset/test',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    # Class balancing
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(train_data.classes),
        y=train_data.classes
    )
    class_weights = dict(enumerate(class_weights))

    return train_data, val_data, test_data, class_weights

# ----------------------
# Model Architecture
# ----------------------
def create_model(input_shape, num_classes):
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    
    # Start with all layers trainable
    base_model.trainable = True

    model = Model(
        inputs=base_model.input,
        outputs=Dense(num_classes, activation='softmax')(base_model.output)
    )
    
    return model

# ----------------------
# Training Pipeline
# ----------------------
def main():
    # Initialize data pipeline
    train_data, val_data, test_data, class_weights = create_data_generators()
    
    # Diagnostic output
    print("\n=== Data Summary ===")
    print(f"Training samples: {train_data.samples}")
    print(f"Validation samples: {val_data.samples}")
    print(f"Test samples: {test_data.samples}")
    print(f"Class weights: {class_weights}")

    # Model configuration
    model = create_model(
        input_shape=(*IMG_SIZE, 3),
        num_classes=len(train_data.class_indices)
    )
    
    model.compile(
        optimizer=Adam(learning_rate=INIT_LR),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Callbacks
    callbacks = [
        ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_accuracy'),
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    ]

    # Training
    print("\n=== Starting Training ===")
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=NUM_EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks
    )

    # Evaluation
    print("\n=== Final Evaluation ===")
    test_loss, test_acc = model.evaluate(test_data)
    print(f"\nTest Accuracy: {test_acc*100:.2f}%")

    # # Mobile conversion
    # converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # tflite_model = converter.convert()
    
    # with open('asl_model.tflite', 'wb') as f:
    #     f.write(tflite_model)
    # print("\nMobile-optimized model saved as 'asl_model.tflite'")

if __name__ == "__main__":
    main()