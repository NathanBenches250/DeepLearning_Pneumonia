import unittest
import tensorflow as tf
from src.model import ChestXrayModel

class TestModel(unittest.TestCase):
    def setUp(self):
        self.model = ChestXrayModel()
    
    def test_model_creation(self):
        model = self.model.build_model()
        
        # Check model type
        self.assertIsInstance(model, tf.keras.Model)
        
        # Check input shape
        self.assertEqual(model.input_shape[1:], (*self.model.image_size, 3))
        
        # Check output shape
        self.assertEqual(model.output_shape[-1], self.model.num_classes)
    
    def test_model_compilation(self):
        model = self.model.build_model()
        
        # Check if model is compiled
        self.assertIsNotNone(model.optimizer)
        self.assertIsNotNone(model.loss)
        self.assertTrue(len(model.metrics) > 0)

if __name__ == '__main__':
    unittest.main()