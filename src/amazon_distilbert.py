from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
CSV_PATH = PROJECT_ROOT / "sentiment_classifier" / "data" / "processed" / "Reviews.csv"

class ReviewDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {key: values[i] for key, values in self.encodings.items()}
        item["labels"] = int(self.labels[i])
        return item


def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0

    for step, batch in enumerate(loader, start=1):
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            outputs = model(**batch)
            loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        if step % 200 == 0:
            print(f"  batch {step}/{len(loader)}  average loss so far {total_loss / step:.4f}")

    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    all_predictions = []

    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                outputs = model(**batch)
            total_loss += outputs.loss.item() * len(batch["labels"])
            all_predictions.append(outputs.logits.argmax(dim=1).cpu().numpy())

    return total_loss / len(loader.dataset), np.concatenate(all_predictions)



if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load your raw CSV with pandas
    df = pd.read_csv(CSV_PATH)  # <-- PUT YOUR CSV FILE PATH HERE

    # Point to your exact text and label columns
    TEXT_COL = "Text"    # <-- CHANGE THIS to your review/text column name
    LABEL_COL = "sentiment"  # <-- CHANGE THIS to your 0/1 target column name

    df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip().str.lower().map({"negative": 0, "positive": 1})
    df = df.dropna(subset=[TEXT_COL, LABEL_COL])
    # df = df.sample(n=100000, random_state=42)  Remove this line later for full 500k

    # Split Data into train and test sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Tokenize the pandas columns
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    train_encodings = tokenizer(train_df[TEXT_COL].astype(str).tolist(), truncation=True, padding=True, max_length=256, return_tensors="pt")
    test_encodings = tokenizer(test_df[TEXT_COL].astype(str).tolist(), truncation=True, padding=True, max_length=256, return_tensors="pt")

    # STEP 5: Feed into YOUR ReviewDataset class
    train_dataset = ReviewDataset(train_encodings, train_df[LABEL_COL].tolist())
    test_dataset = ReviewDataset(test_encodings, test_df[LABEL_COL].tolist())

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # STEP 6: Load model on M5 Pro GPU, train, and print F1/Accuracy report
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2).to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    epochs = 2
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * epochs)

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss, preds = evaluate(model, test_loader, device)
        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    y_true = test_df[LABEL_COL].tolist()
    print(f"\nAccuracy: {accuracy_score(y_true, preds):.4f}")
    print(f"F1 Score: {f1_score(y_true, preds, average='weighted'):.4f}")
    print("\nClassification Report:\n")
    print(classification_report(y_true, preds))