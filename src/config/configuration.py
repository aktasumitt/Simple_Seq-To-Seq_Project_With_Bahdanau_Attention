from src.constant.configs import Configs
from src.constant.params import Params
from src.constant.schema import Schema
from src.entity.config_entity import (DataIngestionConfig,
                                      ModelConfig,
                                      TrainingConfig,
                                      TestConfig,
                                      PredictionConfig)


class Configuration():

    def __init__(self):

        self.config = Configs
        self.params = Params
        self.schema = Schema

    def data_ingestion_config(self):

        configuration = DataIngestionConfig(transformed_train_dataset=self.config.TRANSFORMED_TRAIN_DATASET_PATH,
                                            transformed_test_dataset=self.config.TRANSFORMED_TEST_DATASET_PATH,
                                            transformed_valid_dataset=self.config.TRANSFORMED_VALID_DATASET_PATH,
                                            test_split_rate=self.params.TEST_SPLIT_RATE,
                                            valid_split_rate=self.params.VALID_SPLIT_RATE,
                                            total_data_size=self.params.TOTAL_DATA_SIZE)
        
        return configuration


    def model_config(self):

        configuration = ModelConfig(model_save_path=self.config.MODEL_SAVE_PATH,
                                    channel_size=self.params.CHANNEL_SIZE,
                                    label_size=self.params.LABEL_SIZE,
                                    hidden_size=self.params.HIDDEN_SIZE,
                                    embedding_size=self.params.EMBED_SIZE,
                                    max_len=self.params.max_len,
                                    device=self.params.DEVICE)

        return configuration

    def training_config(self):

        configuration = TrainingConfig(train_dataset_path=self.config.TRANSFORMED_TRAIN_DATASET_PATH,
                                       validation_dataset_path=self.config.TRANSFORMED_VALID_DATASET_PATH,
                                       model_path=self.config.MODEL_SAVE_PATH,
                                       checkpoint_path=self.config.CHECKPOINT_SAVE_PATH,
                                       save_result_path=self.config.SAVE_TRAINING_RESULT_PATH,
                                       final_model_save_path=self.config.FINAL_MODEL_SAVE_PATH,
                                       batch_size=self.params.BATCH_SIZE,
                                       learning_rate=self.params.LEARNING_RATE,
                                       beta1=self.params.BETA1,
                                       beta2=self.params.BETA2,
                                       epochs=self.params.EPOCHS,
                                       device=self.params.DEVICE,
                                       load_checkpoint=self.params.LOAD_CHECKPOINT_FOR_TRAIN
                                       )

        return configuration

    def test_config(self):

        configuration = TestConfig(final_model_path=self.config.FINAL_MODEL_SAVE_PATH,
                                   test_dataset_path=self.config.TRANSFORMED_TEST_DATASET_PATH,
                                   device=self.params.DEVICE,
                                   batch_size=self.params.BATCH_SIZE,
                                   load_checkpoints_for_test=self.params.LOAD_CHECKPOINT_FOR_TEST,
                                   save_tested_model=self.params.SAVE_TESTED_MODEL,
                                   tested_model_save_path=self.config.TESTED_MODEL_SAVE_PATH,
                                   test_result_save_path=self.config.SAVE_TESTING_RESULT_PATH,
                                   best_checkpoints_path=self.config.BEST_CHECKPOINT_PATH
                                   )

        return configuration

    def prediction_config(self):
        
        configuration = PredictionConfig(final_model_path=self.config.FINAL_MODEL_SAVE_PATH,
                                        device=self.params.DEVICE,
                                        batch_size=self.params.BATCH_SIZE,
                                        max_len=self.params.max_len)

        return configuration

