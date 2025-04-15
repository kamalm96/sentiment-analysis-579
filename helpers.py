# Add functions here for pre-processing data

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
import torch
from torch.utils.data import Dataset

# Initialize functions here
stop_words = set(stopwords.words("english"))
stem = PorterStemmer()
lem = WordNetLemmatizer()

# You can use google/chatgpt to help with patterns if needed
url_pattern = re.compile(r"https?://\S+")
# whitespace_pattern = re.compile(r"\s+")
punctuation_pattern = re.compile(r"[^\w\s]")

# helper to map y values to numerical
sentiment_numberical = {"negative": 0, "neutral": 1, "positive": 2}


def remove_urls(text):
    if isinstance(text, str):
        return url_pattern.sub("", text)
    return text


def remove_punctuation(text):
    if isinstance(text, str):
        return punctuation_pattern.sub("", text)
    return text


# This is what im currently using to preprocess our text data, you can add anything to it that you think may be better
def preprocess_text(text, remove_stopwords=True, apply_stem=False, apply_lem=False):
    if isinstance(text, str):
        text = text.lower()
        text = remove_urls(text)
        text = remove_punctuation(text)
        tokens = word_tokenize(text)
        if remove_stopwords:
            tokens = [word for word in tokens if word not in stop_words]

        if apply_stem:
            tokens = [stem.stem(word) for word in tokens]

        if apply_lem:
            tokens = [lem.lemmatize(word) for word in tokens]
        return " ".join(tokens)

    return text


def map_sentiment_to_text(sentiment_value):
    sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}
    return sentiment_map.get(sentiment_value, "unknown")


class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=True,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "token_type_ids": encoding["token_type_ids"].flatten(),
            "label": torch.tensor(label, dtype=torch.long),
        }
