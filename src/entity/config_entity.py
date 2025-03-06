from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataIngestionConfig:

    transformed_train_dataset: Path
    transformed_test_dataset: Path
    transformed_valid_dataset: Path
    test_split_rate: float
    valid_split_rate: float
    total_data_size: int


@dataclass
class ModelConfig:

    model_save_path: Path
    channel_size: int
    label_size: int
    hidden_size: int
    embedding_size: int
    max_len:int
    device:str


@dataclass
class TrainingConfig:

    train_dataset_path: Path
    validation_dataset_path: Path
    model_path: Path
    checkpoint_path: Path
    save_result_path: Path
    final_model_save_path: Path
    batch_size: int
    learning_rate: float
    beta1: float
    beta2: float
    epochs: int
    device: str
    load_checkpoint: bool


@dataclass
class TestConfig:
    final_model_path: Path
    test_dataset_path: Path
    device: str
    batch_size: int
    load_checkpoints_for_test: bool
    save_tested_model: bool
    tested_model_save_path: Path
    test_result_save_path: Path
    best_checkpoints_path: Path


@dataclass
class PredictionConfig:
    final_model_path: Path
    device: str
    batch_size: int
    max_len: Path
