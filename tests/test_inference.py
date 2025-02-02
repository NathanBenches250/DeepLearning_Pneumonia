import unittest
import os
import numpy as np
from src.inference import PneumoniaPredictor

class TestInference(unittest.TestCase):
    def setUp(self):
        self.predictor = PneumoniaPredictor()
    
    def test_prediction(self):
        # Create a sample image for testing
        sample_image_path = "data/raw/test_image.jpg"
        if os.path.exists(sample_image_path):
            prediction, confidence, heatmap = self.predictor.predict(sample_image_path)
            
            # Check prediction type and value
            self.assertIn(prediction, ["Normal", "Pneumonia"])
            
            # Check confidence score
            self.assertIsInstance(confidence, float)
            self.assertTrue(0 <= confidence <= 1)
            
            # Check heatmap
            self.assertIsInstance(heatmap, np.ndarray)
            self.assertEqual(len(heatmap.shape), 3)  # Height, width, channels
    
    def test_heatmap_saving(self):
        test_heatmap = np.random.rand(224, 224, 3)
        save_path = "test_heatmap.jpg"
        
        self.predictor.save_heatmap(test_heatmap, save_path)
        self.assertTrue(os.path.exists(save_path))
        
        # Cleanup
        if os.path.exists(save_path):
            os.remove(save_path)

if __name__ == '__main__':
    unittest.main()