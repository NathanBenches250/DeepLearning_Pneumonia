import os
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.inference import PneumoniaPredictor
from src.config import FLASK_CONFIG, MODEL_DIR

app = Flask(__name__)
app.config.update(FLASK_CONFIG)

# Find the latest model file
model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.keras')]
if model_files:
    latest_model = os.path.join(MODEL_DIR, sorted(model_files)[-1])
    predictor = PneumoniaPredictor(model_path=latest_model)
else:
    predictor = PneumoniaPredictor()  # Fallback to default initialization

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Get prediction
            prediction, confidence, heatmap = predictor.predict(filepath)
            
            if prediction == "Error":
                return jsonify({'error': 'Error processing image'}), 500
            
            # Save heatmap
            heatmap_filename = f'heatmap_{filename}'
            heatmap_path = os.path.join(app.config['UPLOAD_FOLDER'], heatmap_filename)
            
            if predictor.save_heatmap(heatmap, heatmap_path):
                return render_template('result.html',
                                     prediction=prediction,
                                     confidence=confidence * 100,  # Convert to percentage
                                     image_path=f'/static/uploads/{filename}',
                                     heatmap_path=f'/static/uploads/{heatmap_filename}')
            else:
                return render_template('result.html',
                                     prediction=prediction,
                                     confidence=confidence * 100,
                                     image_path=f'/static/uploads/{filename}')
        
        return jsonify({'error': 'Invalid file type'}), 400
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)