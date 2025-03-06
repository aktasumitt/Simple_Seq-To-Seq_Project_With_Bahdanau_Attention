class Configs():
    
    # After data ingestion
    TRANSFORMED_TRAIN_DATASET_PATH = "artifacts/data_transformed/train_dataset.pth"
    TRANSFORMED_TEST_DATASET_PATH = "artifacts/data_transformed/test_dataset.pth"
    TRANSFORMED_VALID_DATASET_PATH = "artifacts/data_transformed/valid_dataset.pth"

    # After creating model
    MODEL_SAVE_PATH = "artifacts/model/model.pth"

    # After training
    CHECKPOINT_SAVE_PATH = "callbacks/checkpoints/checkpoint_last.pth.tar"
    SAVE_TRAINING_RESULT_PATH = "results/train_results.json"
    FINAL_MODEL_SAVE_PATH = "callbacks/final_model/final_model.pth"
    
    # After Testing
    TESTED_MODEL_SAVE_PATH = "callbacks/tested_model/tested_best_model.pth"
    SAVE_TESTING_RESULT_PATH = "results/test_results.json"
    BEST_CHECKPOINT_PATH = "callbacks/checkpoints/checkpoint_5-epoch.pth.tar"  # change this as your results
    
    # After prediction
    SAVE_PREDICTION_RESULT_PATH = "predict_artifact/results/result.json"
    PREDICTION_DATA_PATH= "predict_artifact/images"
    
    