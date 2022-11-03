from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict

import pandas as pd
import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from pandas import DataFrame
from tqdm import tqdm
from sklearn.utils import shuffle

from config import max_input_length, max_target_length, TRAIN_DATASET_PATH, TEST_DATASET_PATH
from utils import modify_tokenizer
from tqdm.notebook import tqdm
import copy

tokenizer = modify_tokenizer()

# def get_dataset():
#     train_path = Path("dataset/train.csv")
#     test_path = Path("dataset/test.csv")

def prepare_dataset() -> None:
    # train_dataset = load_dataset("squad", split="train")
    # test_dataset = load_dataset("squad", split="validation")
    #
    # df_train = get_dataframe(train_dataset)
    # df_test = get_dataframe(test_dataset)
    #
    # save_dataset(df_train, df_test)

    dataset = load_dataset("csv", data_files={"train": str(TRAIN_DATASET_PATH), "test": str(TEST_DATASET_PATH)})
    dataset = dataset.filter(drop_long_inputs, batched=True, load_from_cache_file=False)
    dataset = dataset.map(convert_to_features, batched=True)

    print(dataset["train"][0]["context"])

    dataset = dataset.remove_columns(
        ["context", "answer", "question"]
    )

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    columns = ['input_ids', 'decoder_input_ids', 'attention_mask', 'decoder_attention_mask']
    train_dataset.set_format(type='torch', columns=columns)
    test_dataset.set_format(type='torch', columns=columns)

    torch.save(train_dataset, 'train_data.pt')
    torch.save(test_dataset, 'test_data.pt')


def get_input_encodings(batch, padding: bool = True):
    contexts = [f"context: {context} " for context in batch["context"]]
    answers = [f"answer: {answer} </s>" for answer in batch["answer"]]
    input_batch = [context + answer for context, answer in zip(contexts, answers)]
    input_encodings = tokenizer.batch_encode_plus(input_batch,
                                        max_length=max_input_length,
                                        add_special_tokens=True,
                                        truncation="do_not_truncate",
                                        pad_to_max_length=padding)
    return input_encodings

def convert_to_features(batch):
    input_encodings = get_input_encodings(batch)
    output_batch = [f"question: {question} </s>" for question in batch["question"]]
    target_encodings = tokenizer.batch_encode_plus(output_batch,
                                                   max_length=max_target_length,
                                                   add_special_tokens=True,
                                                   truncation=True, pad_to_max_length=True)
    encodings = {
        'input_ids': input_encodings['input_ids'],
        'attention_mask': input_encodings['attention_mask'],
        'decoder_input_ids': target_encodings['input_ids'],
        'decoder_attention_mask': target_encodings['attention_mask']
    }
    return encodings


def drop_long_inputs(batch) -> List[bool]:
    input_encodings = get_input_encodings(batch, padding=False)
    return [True if len(input_id) <= max_input_length else False for input_id in input_encodings.data["input_ids"]]


def save_dataset(df_train: DataFrame, df_test: DataFrame) -> None:
    train_path = Path("dataset/train.csv")
    test_path = Path("dataset/test.csv")
    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)


def get_dataframe(dataset: Dataset) -> DataFrame:
    df_columns = ["context", "answer", "question"]
    dataframe = DataFrame(columns=df_columns)

    count_long = 0
    count_short = 0

    for idx, text in enumerate(tqdm(dataset)):
        context = text["context"]
        answer = text["answers"]["text"][0]
        question = text["question"]
        num_of_words = len(answer.split())
        print(idx)
        if num_of_words >= 7:
            count_long += 1
            continue
        else:
            dataframe.loc[idx] = [context] + [answer] + [question]
            count_short += 1
    print("count_long train dataset: ", count_long)
    print("count_short train dataset: ", count_short)
    return shuffle(dataframe)


prepare_dataset()
