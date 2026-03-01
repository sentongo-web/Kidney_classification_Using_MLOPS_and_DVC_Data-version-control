import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os


class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        model = load_model(os.path.join("artifacts", "training", "model.h5"))

        img = image.load_img(self.filename, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        result = np.argmax(model.predict(img_array), axis=1)

        if result[0] == 1:
            prediction = "Tumor"
        else:
            prediction = "Normal"

        return [{"image": prediction}]
