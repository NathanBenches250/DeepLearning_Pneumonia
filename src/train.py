import os
import tensorflow as tf
from datetime import datetime
from src.model import ChestXrayModel
from src.data_loader import DataLoader
from src.config import MODEL_CONFIG, MODEL_DIR, TRAIN_CONFIG
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
import numpy as np

def train_model():
    # Initialize data loader and model
    data_loader = DataLoader("data/raw/chest_xray/train")
    model = ChestXrayModel().build_model()
    
    # Create data generators for all splits
    train_data, train_labels = data_loader.get_training_data('/content/chest-xray-pneumonia/data/raw/chest_xray/train')
    test_data, test_labels = data_loader.get_training_data('/content/chest-xray-pneumonia/data/raw/chest_xray/test')
    val_data, val_labels = data_loader.get_training_data('/content/chest-xray-pneumonia/data/raw/chest_xray/val')
    
    x_train = train_data[:]
    y_train = train_labels[:]

    x_test = test_data[:]
    y_test = test_labels[:]

    x_val = val_data[:]
    y_val = val_labels[:]

    x_train = np.array(x_train) / 255
    x_val = np.array(x_val) / 255
    x_test = np.array(x_test) / 255

    x_train = x_train.reshape(-1, MODEL_CONFIG["image_size"][0], MODEL_CONFIG["image_size"][1], 1)
    y_train = np.array(y_train)

    x_val = x_val.reshape(-1, MODEL_CONFIG["image_size"][0], MODEL_CONFIG["image_size"][1], 1)
    y_val = np.array(y_val)

    x_test = x_test.reshape(-1, MODEL_CONFIG["image_size"][0], MODEL_CONFIG["image_size"][1], 1)
    y_test = np.array(y_test)
    
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range = 30,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.2, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip = True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    datagen.fit(x_train)

    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 2, verbose=1,factor=0.3, min_lr=0.000001)

    history = model.fit(datagen.flow(x_train,y_train, batch_size = 32) ,epochs = 12 , validation_data = datagen.flow(x_val, y_val) ,callbacks = [learning_rate_reduction])

    saved_model_path = f"model_checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.keras"

    model.save(saved_model_path)

    return model, history, saved_model_path

if __name__ == "__main__":
    model, history, saved_model_path = train_model()
    print(f"\nModel saved to: {saved_model_path}")