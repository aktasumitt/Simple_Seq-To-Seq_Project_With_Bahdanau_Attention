from src.config.configuration import Configuration
from src.components.prediction.predict import Prediction

class PredictionPipeline():
    def __init__(self):

        configuration=Configuration()
        self.prediction_config=configuration.prediction_config()
    
    def run_prediction_pipeline(self,input_data):
    
        prediction=Prediction(self.prediction_config)
        predict_results=prediction.prediction(input_data)
        
        return predict_results.tolist()
        