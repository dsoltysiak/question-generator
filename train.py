import gc

import torch
from transformers import (
    T5ForConditionalGeneration,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
import wandb
from config import (
    pretrained_model,
    TRAIN_DATASET_TENSOR_PATH,
    TEST_DATASET_TENSOR_PATH,
    MODEL_PATH,
)
from utils import T2TDataCollator


def train() -> None:
    """Training pipeline to fine-tune the pre-trained model"""
    model = T5ForConditionalGeneration.from_pretrained(pretrained_model)

    train_dataset = torch.load(
        TRAIN_DATASET_TENSOR_PATH, map_location=torch.device("cuda")
    )
    test_dataset = torch.load(
        TEST_DATASET_TENSOR_PATH, map_location=torch.device("cuda")
    )

    training_args = TrainingArguments(
        output_dir=str(MODEL_PATH),
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        num_train_epochs=6,
        learning_rate=1e-3,
        logging_steps=300,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=1200,
        load_best_model_at_end=True,
        report_to="wandb",
        metric_for_best_model="eval_loss",
    )
    torch.cuda.empty_cache()
    gc.collect()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=T2TDataCollator(),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    trainer.train()
    wandb.finish()


if __name__ == "__main__":
    train()
