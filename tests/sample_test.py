import unittest
from app import load_and_preprocess_data, train_models
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

class TestIDSModelPipeline(unittest.TestCase):

    def test_data_loading(self):
        """Test if data loads and returns proper shape."""
        X_train, y_train = load_and_preprocess_data()
        self.assertIsNotNone(X_train)
        self.assertIsNotNone(y_train)
        self.assertGreater(X_train.shape[0], 0, "X_train should not be empty")
        self.assertEqual(len(X_train), len(y_train), "X and y should have same number of rows")

    def test_model_training(self):
        """Test if models are trained without throwing errors."""
        X_train, y_train = load_and_preprocess_data()
        try:
            train_models(X_train, y_train)
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"Model training failed with error: {e}")

if __name__ == '__main__':
    unittest.main()
