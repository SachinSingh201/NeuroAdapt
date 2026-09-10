import pickle
import numpy as np
import os
from .image_service import ImageService
from ..config import Config

class LocalEmotionService:
    def __init__(self):
        self.model = None
        self.classes = ["engaged", "bored", "confused", "frustrated"]
        self.load_model()

    def load_model(self):
        """Loads the local emotion classification model."""
        try:
            model_path = Config.LOCAL_MODEL_PATH
            if os.path.exists(model_path):
                with open(model_path, "rb") as file:
                    self.model = pickle.load(file)
                print(f"Local model loaded successfully from {model_path}")
            else:
                print(f"Local model file not found at {model_path}")
        except Exception as e:
            print(f"Error loading local model: {e}")

    def predict(self, image):
        """
        Performs inference on the given image.
        Returns (emotion, confidence)
        """
        if self.model is None:
            return None, 0.0

        try:
            # Preprocess the image
            preprocessed_img = ImageService.preprocess_for_local_model(image)
            if preprocessed_img is None:
                return None, 0.0

            # Add batch dimension
            input_data = np.expand_dims(preprocessed_img, axis=0)

            # Prediction
            # Assuming it's a Keras model as seen in the notebook
            probabilities = self.model.predict(input_data, verbose=0)[0]
            index = np.argmax(probabilities)
            
            emotion = self.classes[index]
            confidence = float(probabilities[index])

            return emotion, confidence
        except Exception as e:
            print(f"Inference error: {e}")
            return None, 0.0
