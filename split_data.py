import os
import shutil
from sklearn.model_selection import train_test_split

dataset_path = "asl_dataset"  # Path to your extracted dataset
train_dir = "dataset/train"
test_dir = "dataset/test"

# Create train/test folders
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

for gesture in os.listdir(dataset_path):
    gesture_path = os.path.join(dataset_path, gesture)
    if not os.path.isdir(gesture_path):
        continue

    # List all images
    images = os.listdir(gesture_path)
    train_imgs, test_imgs = train_test_split(images, test_size=0.2, random_state=42)

    # Copy to train folder
    os.makedirs(os.path.join(train_dir, gesture), exist_ok=True)
    for img in train_imgs:
        src = os.path.join(gesture_path, img)
        dst = os.path.join(train_dir, gesture, img)
        shutil.copy(src, dst)

    # Copy to test folder
    os.makedirs(os.path.join(test_dir, gesture), exist_ok=True)
    for img in test_imgs:
        src = os.path.join(gesture_path, img)
        dst = os.path.join(test_dir, gesture, img)
        shutil.copy(src, dst)