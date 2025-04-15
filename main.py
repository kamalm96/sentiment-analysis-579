import pandas as pd
import numpy as np
import os
import torch
from dotenv import load_dotenv
import praw
import joblib
import nltk
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from helpers import preprocess_text, sentiment_numberical, map_sentiment_to_text
from model import (
    SentimentClassifier,
    load_tokenizer,
    create_data_loaders,
    train_model,
    evaluate_model,
    predict_sentiment,
)

load_dotenv()

# keep those, we need them
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")


def load_and_preprocess_data():
    print("Loading and preprocessing data...")
    train_data = pd.read_csv("train.csv", encoding="ISO-8859-1")
    test_data = pd.read_csv("test.csv", encoding="ISO-8859-1")

    train_data = train_data.dropna(subset=["text", "sentiment"])
    test_data = test_data.dropna(subset=["text", "sentiment"])

    train_data["sentiment_num"] = train_data["sentiment"].map(sentiment_numberical)
    test_data["sentiment_num"] = test_data["sentiment"].map(sentiment_numberical)

    X = train_data["text"].fillna("").values
    y = train_data["sentiment_num"].values
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )

    X_test = test_data["text"].fillna("").values
    y_test = test_data["sentiment_num"].values

    print("Applying text preprocessing...")
    X_train = np.array(
        [preprocess_text(text, remove_stopwords=True) for text in X_train]
    )
    X_val = np.array([preprocess_text(text, remove_stopwords=True) for text in X_val])
    X_test = np.array([preprocess_text(text, remove_stopwords=True) for text in X_test])

    return X_train, y_train, X_val, y_val, X_test, y_test


def train_sentiment_model(X_train, y_train, X_val, y_val, batch_size=16, epochs=3):
    print("\nInitializing BERT model for sentiment analysis...")

    tokenizer = load_tokenizer()

    train_dataloader, val_dataloader = create_data_loaders(
        X_train, y_train, X_val, y_val, tokenizer, batch_size
    )

    model = SentimentClassifier(n_classes=3)

    print("\nTraining model...")
    model = train_model(
        model, train_dataloader, val_dataloader, epochs=epochs, device=device
    )

    model.load_state_dict(torch.load("best_model_state.bin"))

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "classes": 3,
        },
        "sentiment_transformer_model.pth",
    )

    print("\nModel saved as 'sentiment_transformer_model.pth'")

    tokenizer.save_pretrained("./sentiment_tokenizer")
    print("Tokenizer saved to './sentiment_tokenizer'")

    return model, tokenizer


def evaluate_on_test_data(model, tokenizer, X_test, y_test):
    print("\nEvaluating model on test data...")

    batch_size = 32
    all_predictions = []
    all_probabilities = []

    for i in range(0, len(X_test), batch_size):
        end_idx = min(i + batch_size, len(X_test))
        batch_texts = X_test[i:end_idx]

        batch_predictions, batch_probabilities = predict_sentiment(
            model, tokenizer, batch_texts, device
        )
        all_predictions.extend(batch_predictions)
        all_probabilities.extend(batch_probabilities)

    print(f"Test samples: {len(y_test)}, Predictions: {len(all_predictions)}")

    if len(all_predictions) != len(y_test):
        print("WARNING: Prediction count doesn't match test sample count!")
        min_len = min(len(all_predictions), len(y_test))
        all_predictions = all_predictions[:min_len]
        y_test = y_test[:min_len]

    accuracy = accuracy_score(y_test, all_predictions)
    report = classification_report(y_test, all_predictions)

    print(f"Test Accuracy: {accuracy:.4f}")
    print(report)

    confidence_scores = [max(probs) for probs in all_probabilities]
    avg_confidence = sum(confidence_scores) / len(confidence_scores)
    print(f"Average prediction confidence: {avg_confidence:.4f}")

    return all_predictions, all_probabilities


def get_reddit_comments(subreddit_name, limit=100):
    """Fetch comments from a specified subreddit"""
    reddit = praw.Reddit(
        client_id=os.getenv("REDDIT_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        user_agent=os.getenv("REDDIT_USER_AGENT"),
    )

    subreddit = reddit.subreddit(subreddit_name)
    comments = []

    for submission in subreddit.hot(limit=10):
        submission.comments.replace_more(limit=0)
        for comment in submission.comments.list():
            if len(comment.body) > 5:
                comments.append(comment.body)
                if len(comments) >= limit:
                    return comments

    return comments


def analyze_social_media_sentiment(model, tokenizer):
    """Analyze sentiment of social media comments"""
    try:
        subreddits = os.getenv("REDDIT_SUBREDDITS", "news,technology,AskReddit").split(
            ","
        )

        all_results = []
        for subreddit in subreddits:
            print(f"\nFetching comments from r/{subreddit}...")
            comments = get_reddit_comments(subreddit.strip(), limit=100)

            if not comments:
                print(f"No comments found for r/{subreddit}")
                continue

            processed_comments = [
                preprocess_text(comment, remove_stopwords=True) for comment in comments
            ]

            predictions, probabilities = predict_sentiment(
                model, tokenizer, processed_comments, device
            )

            results = []
            for i, (comment, pred, probs) in enumerate(
                zip(comments, predictions, probabilities)
            ):
                confidence = max(probs)
                sentiment_text = map_sentiment_to_text(pred)
                results.append(
                    {
                        "subreddit": subreddit,
                        "comment": (
                            comment[:100] + "..." if len(comment) > 100 else comment
                        ),
                        "sentiment": sentiment_text,
                        "confidence": confidence,
                    }
                )

            all_results.extend(results)

            sentiment_counts = {
                "positive": sum(1 for r in results if r["sentiment"] == "positive"),
                "neutral": sum(1 for r in results if r["sentiment"] == "neutral"),
                "negative": sum(1 for r in results if r["sentiment"] == "negative"),
            }

            avg_confidence = sum(r["confidence"] for r in results) / len(results)

            print(f"\nSentiment analysis for r/{subreddit} ({len(results)} comments):")
            print(
                f"Positive: {sentiment_counts['positive']} ({sentiment_counts['positive']/len(results)*100:.1f}%)"
            )
            print(
                f"Neutral: {sentiment_counts['neutral']} ({sentiment_counts['neutral']/len(results)*100:.1f}%)"
            )
            print(
                f"Negative: {sentiment_counts['negative']} ({sentiment_counts['negative']/len(results)*100:.1f}%)"
            )
            print(f"Average confidence: {avg_confidence:.2f}")

        if all_results:
            results_df = pd.DataFrame(all_results)
            results_df.to_csv("sentiment_analysis_results.csv", index=False)
            print("\nAll results saved to 'sentiment_analysis_results.csv'")

            print("\nSample of analyzed comments:")
            sample_size = min(5, len(all_results))
            sample_indices = np.random.choice(
                len(all_results), sample_size, replace=False
            )
            for idx in sample_indices:
                r = all_results[idx]
                print(
                    f"r/{r['subreddit']} - {r['sentiment']} ({r['confidence']:.2f}): {r['comment']}"
                )

    except Exception as e:
        print(f"Error in social media analysis: {str(e)}")


def main():
    print("Starting sentiment analysis system...")

    X_train, y_train, X_val, y_val, X_test, y_test = load_and_preprocess_data()

    model_path = "sentiment_transformer_model.pth"
    if os.path.exists(model_path) and os.path.exists("./sentiment_tokenizer"):
        print(f"\nLoading existing model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        model = SentimentClassifier(n_classes=checkpoint.get("classes", 3))
        model.load_state_dict(checkpoint["model_state_dict"])
        tokenizer = load_tokenizer()
    else:
        model, tokenizer = train_sentiment_model(X_train, y_train, X_val, y_val)

    evaluate_on_test_data(model, tokenizer, X_test, y_test)

    if os.getenv("REDDIT_CLIENT_ID") and os.getenv("REDDIT_CLIENT_SECRET"):
        print("\n" + "=" * 50)
        print("Testing model on Reddit comments:")
        analyze_social_media_sentiment(model, tokenizer)
    else:
        print(
            "\nSkipping Reddit analysis. Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET in .env file to enable."
        )


if __name__ == "__main__":
    main()
