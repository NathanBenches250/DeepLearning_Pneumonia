import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.config import MODEL_CONFIG, TRAIN_CONFIG, AUGMENTATION_CONFIG

class DataLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_size = MODEL_CONFIG["image_size"]
        self.labels = ['PNEUMONIA', 'NORMAL']
        self.augmentation = ImageDataGenerator(**AUGMENTATION_CONFIG)
        
    def load_and_preprocess_image(self, image_path):
        """Load and preprocess a single image."""
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.image_size)
        image = image / 255.0  # Normalize
        return image

    def load_dataset(self):
        """Load and split the dataset."""
        images = []
        labels = []
        classes = ["NORMAL", "PNEUMONIA"]
        
        for class_idx, class_name in enumerate(classes):
            class_path = os.path.join(self.data_dir, class_name)
            for image_name in os.listdir(class_path):
                image_path = os.path.join(class_path, image_name)
                image = self.load_and_preprocess_image(image_path)
                images.append(image)
                labels.append(class_idx)
        
        images = np.array(images)
        labels = np.array(labels)
        
        # Split dataset
        X_train, X_temp, y_train, y_temp = train_test_split(
            images, labels,
            test_size=(TRAIN_CONFIG["val_split"] + TRAIN_CONFIG["test_split"]),
            random_state=TRAIN_CONFIG["random_seed"]
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=TRAIN_CONFIG["test_split"]/(TRAIN_CONFIG["val_split"] + TRAIN_CONFIG["test_split"]),
            random_state=TRAIN_CONFIG["random_seed"]
        )
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def get_training_data(self, data_dir):
        data = [] 
        for label in self.labels: 
            path = os.path.join(data_dir, label)
            class_num = self.labels.index(label)
            for img in os.listdir(path):
                try:
                    img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                    resized_arr = cv2.resize(img_arr, MODEL_CONFIG["image_size"]) # Reshaping images to preferred size
                    data.append([resized_arr, class_num])
                except Exception as e:
                    print(e)
        return np.array(data)