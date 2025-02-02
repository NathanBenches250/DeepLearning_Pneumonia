import tensorflow as tf
from transformers import TFAutoModelForImageClassification
from src.config import MODEL_CONFIG
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout , BatchNormalization

class ChestXrayModel:
    def __init__(self):
        self.model_name = MODEL_CONFIG["model_name"]
        self.num_classes = MODEL_CONFIG["num_classes"]
        self.image_size = MODEL_CONFIG["image_size"]
        
    def build_model(self):
        model = Sequential()
        model.add(Conv2D(32 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = (150,150,1)))
        model.add(BatchNormalization())
        model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
        model.add(Conv2D(64 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
        model.add(Dropout(0.1))
        model.add(BatchNormalization())
        model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
        model.add(Conv2D(64 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
        model.add(BatchNormalization())
        model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
        model.add(Conv2D(128 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
        model.add(Conv2D(256 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
        model.add(Flatten())
        model.add(Dense(units = 128 , activation = 'relu'))
        model.add(Dropout(0.2))
        model.add(Dense(units = 1 , activation = 'sigmoid'))
        model.compile(optimizer = "rmsprop" , loss = 'binary_crossentropy' , metrics = ['accuracy'])
        return model
    
    def load_model(self, model_path):
        return tf.keras.models.load_model(model_path)