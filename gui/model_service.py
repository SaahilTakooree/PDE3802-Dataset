# Import the neccessary dependency.
import tensorflow as tf



# Define a service class to handle the model loading and prediction.
class ModelService:
    
    # Intialised the model service by loading the model.
    def __init__(self, model_path="best_model.h5"):
        print("Loading model. Please wait.")
        self.model = tf.keras.models.load_model(model_path)
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


    # Predicts the input image using the loade model.
    def predict_image(self, image):
        predictions = self.model.predict(image)
        class_index = predictions.argmax()
        confidence = predictions.max()
        return predictions, class_index, confidence