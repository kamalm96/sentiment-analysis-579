# Add functions here for pre-processing data

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer

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


# If needed later, we could use this


# def remove_whitespace(text):
#     if isinstance(text, str):
#         return whitespace_pattern.sub("", text)
#     return text


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
