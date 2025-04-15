import torch
import os
from model import SentimentClassifier, load_tokenizer, predict_sentiment
from helpers import preprocess_text, map_sentiment_to_text

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

model_path = "sentiment_transformer_model.pth"
checkpoint = torch.load(model_path, map_location=device)
model = SentimentClassifier(n_classes=checkpoint.get("classes", 3))
model.load_state_dict(checkpoint["model_state_dict"])
tokenizer = load_tokenizer()


def analyze_text(text):
    processed_text = preprocess_text(text, remove_stopwords=True)
    predictions, probabilities = predict_sentiment(
        model, tokenizer, [processed_text], device
    )
    sentiment = map_sentiment_to_text(predictions[0])
    confidence = max(probabilities[0])
    return sentiment, confidence


if __name__ == "__main__":
    sample_texts = [
        "I absolutely love this product, it works great!",
        "The service was terrible and the staff was rude.",
        "The movie was okay, nothing special but not bad either.",
    ]

    for text in sample_texts:
        sentiment, confidence = analyze_text(text)
        print(f"Text: {text}")
        print(f"Sentiment: {sentiment}")
        print(f"Confidence: {confidence:.4f}")
        print("-" * 50)

    print("Enter text to analyze (or 'quit' to exit):")
    while True:
        user_input = input("> ")
        if user_input.lower() == "quit":
            break
        sentiment, confidence = analyze_text(user_input)
        print(f"Sentiment: {sentiment} (Confidence: {confidence:.4f})")
