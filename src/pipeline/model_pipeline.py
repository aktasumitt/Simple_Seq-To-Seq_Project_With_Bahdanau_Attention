from src.config.configuration import Configuration
from src.components.model.model import create_and_save_model


class ModelPipeline():
    def __init__(self):

        configuration = Configuration()
        self.modelconfig = configuration.model_config()

    def run_model_creating(self):

        create_and_save_model(self.modelconfig)


if __name__=="__main__":
    
    model_pipeline=ModelPipeline()
    model_pipeline.run_model_creating()