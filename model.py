import torch
import torch.nn as nn
from transformers import (
    BertModel,
    BertTokenizer,
    AdamW,
    get_linear_schedule_with_warmup,
)
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
import random
import time
from helpers import SentimentDataset


def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


set_seed(42)


class SentimentClassifier(nn.Module):
    def __init__(self, n_classes=3):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.drop = nn.Dropout(p=0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled_output = outputs.pooler_output
        output = self.drop(pooled_output)
        return self.fc(output)


def train_model(model, train_data_loader, val_data_loader, epochs=4, device="cuda"):
    if not torch.cuda.is_available():
        device = "cpu"
        print(f"GPU not available, using CPU instead.")
    else:
        print(f"Using {device} for training.")

    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

    total_steps = len(train_data_loader) * epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    best_val_accuracy = 0

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 10)

        t0 = time.time()
        total_loss = 0
        model.train()

        for batch in train_data_loader:
            model.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask, token_type_ids)

            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()

            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            scheduler.step()

        avg_train_loss = total_loss / len(train_data_loader)
        print(f"Average training loss: {avg_train_loss:.4f}")
        print(f"Training epoch took: {(time.time() - t0):.2f}s")

        val_accuracy, val_report = evaluate_model(model, val_data_loader, device)
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print(val_report)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), "best_model_state.bin")
            print("Saved best model!")

    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    return model


def evaluate_model(model, data_loader, device):
    model.eval()

    predictions = []
    actual_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask, token_type_ids)

            _, preds = torch.max(outputs, dim=1)

            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())

    accuracy = accuracy_score(actual_labels, predictions)

    report = classification_report(actual_labels, predictions)

    return accuracy, report


def predict_sentiment(model, tokenizer, texts, device="cuda"):
    if not torch.cuda.is_available():
        device = "cpu"

    model = model.to(device)
    model.eval()

    all_predictions = []
    all_probabilities = []

    if not isinstance(texts, list):
        texts = [str(texts)]
    else:
        texts = [str(text) if text is not None else "" for text in texts]

    batch_size = 16
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]

        encodings = tokenizer(
            batch_texts,
            add_special_tokens=True,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)
        token_type_ids = encodings["token_type_ids"].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask, token_type_ids)

        probs = torch.nn.functional.softmax(outputs, dim=1)

        confidence_values, predictions = torch.max(probs, dim=1)

        all_predictions.extend(predictions.cpu().tolist())
        all_probabilities.extend(probs.cpu().tolist())

    return all_predictions, all_probabilities


def load_tokenizer():
    return BertTokenizer.from_pretrained("bert-base-uncased")


def create_data_loaders(X_train, y_train, X_val, y_val, tokenizer, batch_size=16):
    train_dataset = SentimentDataset(X_train, y_train, tokenizer)
    val_dataset = SentimentDataset(X_val, y_val, tokenizer)

    train_sampler = RandomSampler(train_dataset)
    val_sampler = SequentialSampler(val_dataset)

    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=batch_size
    )

    val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=batch_size)

    return train_dataloader, val_dataloader
