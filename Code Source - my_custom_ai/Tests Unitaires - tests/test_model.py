import unittest
import numpy as np
from my_custom_ai.model import MyCustomAI

class TestMyCustomAI(unittest.TestCase):

    def setUp(self):
        self.model = MyCustomAI()
        self.X = np.random.rand(100, 10)
        self.y = np.random.randint(0, 2, size=100)
    
    def test_train(self):
        self.model.train(self.X, self.y)
    
    def test_predict(self):
        self.model.train(self.X, self.y)
        new_data = np.random.rand(5, 10)
        predictions = self.model.predict(new_data)
        self.assertEqual(predictions.shape[0], 5)
    
    def test_save_and_load_model(self):
        self.model.train(self.X, self.y)
        self.model.save_model("test_model.joblib")
        self.model.load_model("test_model.joblib")

if __name__ == '__main__':
    unittest.main()
