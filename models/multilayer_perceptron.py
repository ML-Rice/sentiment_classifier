import numpy as np
# I used PyTorch for this, so may have to put it into project dependecies since I didnt see it loaded with uv sync
import torch
from torch import nn
from sklearn.metrics import accuracy_score, f1_score

def build_model(input_size, num_layers, hidden_size=256):
    layers = []
    current_size = input_size

    for i in range(num_layers):
        layers.append(nn.Linear(current_size, hidden_size))
        layers.append(nn.ReLU())
        current_size = hidden_size

    layers.append(nn.Linear(current_size, 1))
    return nn.Sequential(*layers)

def to_tensor(sparse_rows, device):
    return torch.tensor(sparse_rows.toarray(), dtype=torch.float32).to(device)

def train_epoch(model, X, y, optimizer, loss_fn, device, batch_size=256):
    model.train()
    order = np.random.permutation(X.shape[0])
    total_loss = 0.0
    num_batches = 0

    for start in range(0, X.shape[0], batch_size):
        batch_rows = order[start:start + batch_size]
        X_batch = to_tensor(X[batch_rows], device)
        y_batch = torch.tensor(y[batch_rows], dtype=torch.float32).to(device)

        scores = model(X_batch).squeeze(1)
        loss = loss_fn(scores, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
    return total_loss/num_batches

def evaluate(model, X, y, loss_fn, device, batch_size=1024):
    model.eval()
    total_loss = 0.0
    all_predictions = []

    with torch.no_grad():
        for start in range(0, X.shape[0], batch_size):
            X_batch = to_tensor(X[start:start + batch_size], device)
            y_batch = torch.tensor(y[start:start + batch_size], dtype=torch.float32).to(device)

            scores = model(X_batch).squeeze(1)
            total_loss += loss_fn(scores, y_batch).item() * len(y_batch)
            all_predictions.append((scores > 0).int().cpu().numpy())

    return total_loss / X.shape[0], np.concatenate(all_predictions)

def train_and_validate(num_layers, X_train, y_train, X_val, y_val, device,
                       num_epochs=5, learning_rate=0.001, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = build_model(X_train.shape[1], num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()

    history = []
    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(model, X_train, y_train, optimizer, loss_fn, device)
        val_loss, val_predictions = evaluate(model, X_val, y_val, loss_fn, device)
        val_accuracy = accuracy_score(y_val, val_predictions)
        val_f1 = f1_score(y_val, val_predictions)

        history.append({
            "num_layers": num_layers,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "val_f1": val_f1,
        })
        print(f"layers {num_layers}  epoch {epoch}  train loss {train_loss:.4f}  "
              f"val loss {val_loss:.4f}  val acc {val_accuracy:.3f}  val F1 {val_f1:.3f}")

    return history