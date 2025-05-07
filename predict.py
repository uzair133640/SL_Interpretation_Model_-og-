import os
import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = all messages, 2 = only errors

# Configuration
MODEL_PATH = '/content/SL_Interpretation_Model/best_model.h5'
TEST_DATA_DIR = '/content/SL_Interpretation_Model/dataset/test'
CLASS_ORDER = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
               'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
               ]  # Manual ordering

def main():
    # Load model
    model = load_model(MODEL_PATH)
    
    # Verify input shape
    input_shape = model.input_shape[1:3]  # (height, width)
    print(f"Model expects input size: {input_shape}")
    
    # Create class mapping
    class_indices = {class_name: idx for idx, class_name in enumerate(CLASS_ORDER)}
    idx_to_class = {v: k for k, v in class_indices.items()}
    
    # Find all test images
    image_paths = []
    for class_name in CLASS_ORDER:
        class_dir = os.path.join(TEST_DATA_DIR, class_name)
        if os.path.exists(class_dir):
            images = [os.path.join(class_dir, f) for f in os.listdir(class_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            image_paths.extend(images[:2])  # Test first 2 images per class

    if not image_paths:
        raise ValueError("No test images found!")

    # Process images
    for img_path in image_paths:
        # Get true class from path
        true_class = os.path.basename(os.path.dirname(img_path))
        
        # Load and preprocess image
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Could not read: {img_path}")
            continue
            
        # Resize and normalize
        img = cv2.resize(img, input_shape)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        img = img / 255.0
        img = np.expand_dims(img, axis=0)  # Add batch dimension

        # Predict
        try:
            preds = model.predict(img)
            pred_idx = np.argmax(preds)
            confidence = np.max(preds)
            predicted_class = idx_to_class[pred_idx]
            
            print(f"\nImage: {os.path.basename(img_path)}")
            print(f"True class: {true_class.upper()}")
            print(f"Predicted: {predicted_class.upper()} ({confidence:.2%})")
            print("------------------------")
            
        except Exception as e:
            print(f"❌ Error processing {img_path}: {str(e)}")

if __name__ == "__main__":
    main()