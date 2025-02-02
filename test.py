import cv2
import numpy as np
from tensorflow.keras.models import load_model

def preprocess_image(image_path):
    img_size = 140
    try:
        img_arr = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        resized_arr = cv2.resize(img_arr, (img_size, img_size))
        resized_arr = resized_arr.reshape(-1, img_size, img_size, 1)
        resized_arr = np.array(resized_arr) / 255
        return resized_arr
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

# Load the saved model
model = load_model('models/saved_models/my_model.keras')

# Paths to the two images for testing
image_paths = [
    'data\\testing\\NORMAL2-IM-1401-0001.jpeg',
    'data\\testing\\NORMAL2-IM-1406-0001.jpeg',
    'data\\testing\\NORMAL2-IM-1412-0001.jpeg',
    'data\\testing\\NORMAL2-IM-1419-0001.jpeg',
    'data\\testing\\NORMAL2-IM-1422-0001.jpeg',
    'data\\testing\\person82_virus_154.jpeg',
    'data\\testing\\person1216_virus_2062.jpeg',
    'data\\testing\\person1229_virus_2080.jpeg',
    'data\\testing\\person1309_bacteria_3294.jpeg',
    'data\\testing\\person1396_bacteria_3545.jpeg'
]

# Preprocess and predict for each image
for image_path in image_paths:
    preprocessed_image = preprocess_image(image_path)
    if preprocessed_image is not None:
        prediction = (model.predict(preprocessed_image) > 0.5).astype("int32")
        print(f"Prediction for {image_path}: {prediction}")
    else:
        print("Skipped prediction due to error.")