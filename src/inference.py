import cv2
import numpy as np
import tensorflow as tf
from src.model import ChestXrayModel
from src.data_loader import DataLoader
from sklearn.model_selection import ShuffleSplit
from src.visualization import generate_gradcam
from src.config import MODEL_CONFIG
from fastai.vision.all import PILImage
from fastai.vision.all import cnn_learner, resnet50
from fastai.vision.all import *
import pickle
import os

class PneumoniaPredictor:
    def __init__(self, model_path=None):
        self.model = ChestXrayModel()
        if model_path and os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
        else:
            self.model = self.model.build_model()


        classifier_model_path = "../classifier_model/chest_xray_classifier.pth"
        if os.path.exists(classifier_model_path):
            print(f"Model found at {classifier_model_path}")
        else:
            print(f"Model not found at {classifier_model_path}")
                
        # Load the chest X-ray classifier model
        if classifier_model_path and os.path.exists(classifier_model_path):
            # Load the learner object from the saved model
            INPUT_PATH = "../archive"
            df_train = pd.read_csv(Path(INPUT_PATH, "train_df.csv"))
            df_train['Target'] = df_train['Target'].apply(lambda x: x.strip())
            df_train['Target'] = df_train['Target'].apply(lambda x: '1' if '3' in x.split() else '0')

            X = df_train
            y = df_train['Target']
            sss = ShuffleSplit(n_splits=1, test_size=.2, random_state=42)
            train_idx, val_idx = next(sss.split(X, y))
            df_train['is_valid'] = False
            df_train.loc[val_idx, 'is_valid'] = True
            dls = ImageDataLoaders.from_df(df_train, 
                               fn_col='image_path',
                               label_col='Target',
                               label_delim=' ',
                               valid_col='is_valid',
                               path=INPUT_PATH,
                               item_tfms=Resize(224),
                               batch_tfms=aug_transforms(size=224, min_scale=0.75))  # Avoid perspective augmentations

            f1_macro = F1ScoreMulti(thresh=0.5, average='macro')
            f1_samples = F1ScoreMulti(thresh=0.5, average='samples')
            f1_micro = F1ScoreMulti(thresh=0.5, average='micro')
            f1_weighted = F1ScoreMulti(thresh=0.5, average='weighted')
            self.learn = cnn_learner(dls, resnet50, metrics=[partial(accuracy_multi, thresh=0.5), f1_macro, f1_samples, f1_micro, f1_weighted])

            # Load the weights into the model
            classifier_model_path = classifier_model_path.rstrip('.pth')
            self.learn.load(classifier_model_path)  

        else:
            raise ValueError("Classifier model path is invalid or not provided.")

        self.img_size = MODEL_CONFIG["image_size"][0]
        self.classes = ["PNEUMONIA", "NORMAL"]
    
    def preprocess_image(self, image_path):
        """Preprocess image using the same method as test.py"""
        try:
            img_arr = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            resized_arr = cv2.resize(img_arr, (self.img_size, self.img_size))
            resized_arr = resized_arr.reshape(-1, self.img_size, self.img_size, 1)
            resized_arr = np.array(resized_arr) / 255
            return resized_arr
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None
    
    def predict(self, image_path):
        """
        Predict pneumonia from an X-ray image.
        Returns: prediction label, confidence score, and activation map
        """
        try:
            # Preprocess image
            preprocessed_image = self.preprocess_image(image_path)
            if preprocessed_image is None:
                return "Error", 0.0, None
            
            # Check if the image is a chest X-ray
            img = PILImage.create(image_path)
            pred, _, probs = self.learn.predict(img)
            print(pred)
            if pred == ['0']:
                return "Error: Please upload a valid chest X-ray image.", 0.0, None

            # Get prediction
            prediction = self.model.predict(preprocessed_image, verbose=0)
            predicted_class = (prediction > 0.5).astype("int32")[0][0]
            confidence = float(prediction[0][0]) if predicted_class == 1 else float(1 - prediction[0][0])
            
            # Generate activation map
            heatmap = generate_gradcam(self.model, preprocessed_image, predicted_class)
            if heatmap is None:
                heatmap = np.zeros_like(preprocessed_image[0])
            
            return self.classes[predicted_class], confidence, heatmap
            
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return "Error", 0.0, None
    
    @staticmethod
    def save_heatmap(heatmap, save_path):
        """Save the activation map visualization."""
        try:
            if heatmap is not None:
                cv2.imwrite(save_path, heatmap)
            return True
        except Exception as e:
            print(f"Error saving heatmap: {str(e)}")
            return False