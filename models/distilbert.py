from pathlib import Path
 
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)
 
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 512          # IMPORTANT: even with 512 tokens, like about 13% of reviews are truncated
VALIDATION_SIZE = 0.2 
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 64
LEARNING_RATE = 2e-5
NUM_EPOCHS = 2
WARMUP_FRACTION = 0.1
 
 
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
 
 
def load_config():
    with (PROJECT_ROOT / "src" / "config.yaml").open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)
 
 
def load_and_split_data(config):
    dataset_path = PROJECT_ROOT / config["dataset_path"].replace("\\", "/")
    df = pd.read_csv(dataset_path)
    df["review"] = df["review"].str.replace("<br />", " ", regex=False)
    df["label"] = (df["sentiment"] == "positive").astype(int)
 
    seed = int(config["random_seed"])
    train_full_df, test_df = train_test_split(
        df, test_size=float(config["test_size"]), random_state=seed, stratify=df["sentiment"]
    )
    train_df, val_df = train_test_split(
        train_full_df, test_size=VALIDATION_SIZE, random_state=seed, stratify=train_full_df["label"]
    )
    return train_df, val_df, test_df
 
 
def make_loader(df, tokenizer, batch_size, shuffle):
    encodings = tokenizer(df["review"].tolist(), truncation=True, max_length=MAX_LENGTH)
    dataset = ReviewDataset(encodings, df["label"].to_numpy())
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=DataCollatorWithPadding(tokenizer)
    )
 
 
def train_epoch(model, loader, optimizer, scheduler, scaler, device):
    model.train()
    total_loss = 0.0
 
    for step, batch in enumerate(loader, start=1):
        batch = {key: value.to(device) for key, value in batch.items()}
 
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
            outputs = model(**batch)            # 1. forward pass
            loss = outputs.loss                 # 2. loss (the model calculates it)
 
        optimizer.zero_grad()                   # 3. clear old gradients
        scaler.scale(loss).backward()           # 4. gradients, scaled up for 16-bit math
        scaler.step(optimizer)                  # 5. scale back down, then nudge the weights
        scaler.update()                         #    adjust the scaling factor for next time
        scheduler.step()                        #    move the learning rate along its schedule
 
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
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
                outputs = model(**batch)
            total_loss += outputs.loss.item() * len(batch["labels"])
            all_predictions.append(outputs.logits.argmax(dim=1).cpu().numpy())
 
    return total_loss / len(loader.dataset), np.concatenate(all_predictions)
 
 
def predict_texts(model, tokenizer, texts, device, max_length=MAX_LENGTH):
    model.eval()
    encoded = tokenizer(
        texts, truncation=True, max_length=max_length, padding=True, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        scores = model(**encoded).logits
    return torch.softmax(scores, dim=1)[:, 1].cpu().numpy()
 
 
def fine_tune(model, train_loader, val_loader, val_labels, device, num_epochs=NUM_EPOCHS):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = num_epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(WARMUP_FRACTION * total_steps), num_training_steps=total_steps
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
 
    best_val_loss = float("inf")
    best_weights = None
 
    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}")
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, scaler, device)
        val_loss, val_predictions = evaluate(model, val_loader, device)
        print(f"epoch {epoch}  train loss {train_loss:.4f}  val loss {val_loss:.4f}  "
              f"val acc {accuracy_score(val_labels, val_predictions):.3f}  "
              f"val F1 {f1_score(val_labels, val_predictions):.3f}")
 
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            print("New best model saved :)")
 
    model.load_state_dict(best_weights)
    print(f"Loaded best model (val loss {best_val_loss:.4f}) :)")
    return model
 
 
def main():
    config = load_config()
    seed = int(config["random_seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
 
    train_df, val_df, test_df = load_and_split_data(config)
    print(f"train {len(train_df)}  validation {len(val_df)}  test {len(test_df)}")
 
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_loader = make_loader(train_df, tokenizer, TRAIN_BATCH_SIZE, shuffle=True)
    val_loader = make_loader(val_df, tokenizer, EVAL_BATCH_SIZE, shuffle=False)
    test_loader = make_loader(test_df, tokenizer, EVAL_BATCH_SIZE, shuffle=False)
 
    torch.manual_seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)
    model = fine_tune(model, train_loader, val_loader, val_df["label"].to_numpy(), device)
 
    _, test_predictions = evaluate(model, test_loader, device)
    test_labels = test_df["label"].to_numpy()
    print(f"Test accuracy:  {accuracy_score(test_labels, test_predictions):.3f}")
    print(f"Test precision: {precision_score(test_labels, test_predictions):.3f}")
    print(f"Test recall:    {recall_score(test_labels, test_predictions):.3f}")
    print(f"Test F1:        {f1_score(test_labels, test_predictions):.3f}")
 
    output_dir = PROJECT_ROOT / "models" / "distilbert_finetuned"
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved fine-tuned model to {output_dir}")
 
 
if __name__ == "__main__":
    main()