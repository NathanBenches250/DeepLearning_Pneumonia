#  Pneumonia Detection Project

A web-based deep learning project for diagnosing pneumonia from chest X-ray images.

## Installation

1. Create a virtual environment:
"""
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
"""

2. Install dependencies:
"""
pip install -r requirements.txt
"""
wawa
3. Download the dataset:
- Visit https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
- Download and extract to `data/raw` directory

4. Download the classifier dataset
- Visit [X-Ray Body Images in PNG (UNIFESP) Competition Dataset](https://www.kaggle.com/datasets/ibombonato/xray-body-images-in-png-unifesp-competion)
- Create a directory named `archive` and download the dataset into that directory.

## Usage

1. Train the model:
"""
python -m src.train
"""
Extract it to models/saved_models/ folder. The pretrained model is present in the folder currently.

2.There is a classifier_model folder that contains a Jupyter notebook. Run the notebook, and save the trained model in the folder itself. The notebook will guide you through the training process.

3. Run the web application:
"""
python -m webapp.app
"""

4. Open your browser and navigate to `http://localhost:5000`













