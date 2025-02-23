import pandas as pd
from helpers import preprocess_text, sentiment_numberical
import nltk
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# keep those, we need them
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# defining our and train and test data here, you can add more data if you want and preprocess it using functions in helpers.py
train_data = pd.read_csv("train.csv", encoding="ISO-8859-1")
test_data = pd.read_csv("test.csv", encoding="ISO-8859-1")

train_data = train_data.dropna(subset=["text", "sentiment"])
test_data = test_data.dropna(subset=["text", "sentiment"])

X_train = train_data["text"].fillna("")
y_train = train_data["sentiment"].map(sentiment_numberical)

X_test = test_data["text"].fillna("")
y_test = test_data["sentiment"].map(sentiment_numberical)

X_train = X_train.apply(lambda x: preprocess_text(x, apply_stem=True))
X_test = X_test.apply(lambda x: preprocess_text(x, apply_stem=True))

pipe = ImbPipeline(
    [
        (
            "vec",
            CountVectorizer(
                stop_words="english", min_df=5, max_df=0.9, ngram_range=(1, 2)
            ),
        ),
        ("tfid", TfidfTransformer()),
        ("lr", LogisticRegression(class_weight="balanced", max_iter=500)),
    ]
)

model = pipe.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
