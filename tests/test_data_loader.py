import unittest
import os
import numpy as np
from src.data_loader import DataLoader

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        self.data_loader = DataLoader("data/raw")
        
    def test_load_and_preprocess_image(self):
        # Create a sample image for testing
        sample_image_path = "data/raw/test_image.jpg"
        if os.path.exists(sample_image_path):
            image = self.data_loader.load_and_preprocess_image(sample_image_path)
            
            # Check image shape and normalization
            self.assertEqual(len(image.shape), 3)
            self.assertEqual(image.shape[:2], self.data_loader.image_size)
            self.assertTrue(np.all(image >= 0) and np.all(image <= 1))
    
    def test_dataset_split(self):
        if os.path.exists("data/raw"):
            (X_train, y_train), (X_val, y_val), (X_test, y_test) = self.data_loader.load_dataset()
            
            # Check that splits sum to total dataset
            total_samples = len(X_train) + len(X_val) + len(X_test)
            self.assertGreater(total_samples, 0)
            
            # Check label consistency
            self.assertEqual(len(X_train), len(y_train))
            self.assertEqual(len(X_val), len(y_val))
            self.assertEqual(len(X_test), len(y_test))

if __name__ == '__main__':
    unittest.main()