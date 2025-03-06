class Params():
    
    # For data ingestion
    TEST_SPLIT_RATE = 0.1
    VALID_SPLIT_RATE = 0.2
    TOTAL_DATA_SIZE=10000

    # For Model
    CHANNEL_SIZE = 1
    LABEL_SIZE = 10
    HIDDEN_SIZE=512
    EMBED_SIZE=512
    max_len=8
    
    # For Training
    BATCH_SIZE = 100
    LEARNING_RATE = 0.001
    BETA1 = 0.9
    BETA2 = 0.98
    EPOCHS = 7
    DEVICE = "cuda"
    LOAD_CHECKPOINT_FOR_TRAIN=False
    

    # For Testing
    LOAD_CHECKPOINT_FOR_TEST=False
    SAVE_TESTED_MODEL=False