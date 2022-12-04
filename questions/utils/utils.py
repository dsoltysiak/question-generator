from typing import List, Any
import logging

import torch
from keybert import KeyBERT
from nltk import sent_tokenize
from transformers import T5Tokenizer, T5TokenizerFast

from config import (
    max_input_length,
    max_target_length,
    pretrained_model,
)
from questgen import settings

logger = logging.getLogger(__name__)


def get_questions(text: str) -> str:
    keyword = get_best_keywords(text)
    question = get_question(text, keyword)
    print(keyword)
    return question.split(":")[-1].strip()


def get_question(text: str, keyword: str) -> str:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = getattr(settings, "QUESTION_MODEL")
    tokenizer = T5TokenizerFast.from_pretrained(pretrained_model)
    input = f"ask question: {text} answer: {keyword} </s>"
    encoding = tokenizer.encode_plus(
        input, max_length=max_input_length, padding=True, return_tensors="pt"
    )
    input_ids, attention_mask = encoding["input_ids"].to(device), encoding[
        "attention_mask"
    ].to(device)
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_target_length,
        early_stopping=True,
        num_beams=5,
        num_return_sequences=1,
    )
    question = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return question.strip()


def get_best_keywords(text: str) -> str:
    keywords = get_keywords(text)
    summarized_text = summarize_text(text)
    keywords_summarized = get_keywords(summarized_text)
    important_keywords = [
        keyword for keyword in keywords if keyword in keywords_summarized
    ]
    logging.info(f"Keywords: {important_keywords}")
    return important_keywords[0]


def get_keywords(text: str) -> tuple[Any]:
    keywords_model = KeyBERT()
    keywords = keywords_model.extract_keywords(text)
    return list(zip(*keywords))[0]


def summarize_text(text: str) -> str:
    model = getattr(settings, "SUMMARY_MODEL")
    tokenizer = T5Tokenizer.from_pretrained(pretrained_model)
    text = text.strip().replace("\n", " ")
    text = "summarize: " + text
    encoding = tokenizer.encode_plus(
        text,
        max_length=max_input_length,
        pad_to_max_length=False,
        truncation=True,
        return_tensors="pt",
    ).to("cuda")

    input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

    outs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        early_stopping=True,
        num_beams=4,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        min_length=8,
        max_length=64,
    ).to("cuda")

    summary = tokenizer.decode(outs[0], skip_special_tokens=True)
    summary = postprocess_text(summary)
    summary = summary.strip()
    logging.info(f"Summarized text: {summary}")
    return summary


def postprocess_text(text: str) -> str:
    final_text = ""
    for sent in sent_tokenize(text):
        sent = sent.capitalize()
        final_text += " " + sent
    return final_text
