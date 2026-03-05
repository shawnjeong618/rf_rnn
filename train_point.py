import random, json, os, copy
from rf_rnn import RFRNN
import torch
from data_loader import CSVSequenceDataset
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd
import numpy as np

# architecture variables
hidden_size = 8
seq_len = 100
output_size = 2

# training variables
SEED = 0            # determinism
BATCH_SIZE = 128
EPOCHS = 200

# deterministic CUDA
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# load dataset
manifest = "manifest.csv"
label_map = {"normal":0, "point":1}
full_ds = CSVSequenceDataset(manifest, seq_len, label_map=label_map)
n = len(full_ds)
n_train = int(0.8 * n)
n_test = n - n_train

# spilt train and test dataset
train_ds, test_ds = random_split(full_ds, [n_train, n_test], generator=torch.Generator().manual_seed(SEED))
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)


# define model parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device = {device}")
model = RFRNN(hidden_size, seq_len, output_size).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
crit = nn.CrossEntropyLoss()

# functions
def accuracy_from_logits(logits, targets):
    return (logits.argmax(1) == targets).float().mean().item()

@torch.no_grad
def evaluate(loader, model):
    model.eval()
    tot_loss, tot_acc, tot = 0.0, 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = crit(logits, y)
        b = x.size(0)
        tot_loss += loss.item() * b
        tot_acc += accuracy_from_logits(logits, y) * b
        tot += b
    return tot_loss / tot, tot_acc / tot

def train_one_epoch():
    model.train()
    tot_loss, tot_acc, tot = 0.0, 0.0, 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = crit(logits, y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        opt.step()
        b = x.size(0)
        tot_loss += loss.item() * b
        tot_acc += accuracy_from_logits(logits, y) * b
        tot += b
    return tot_loss / tot, tot_acc / tot

history = []
best_loss = float('inf')
for ep in range(1, EPOCHS+1):
    tr_loss, tr_acc = train_one_epoch()
    va_loss, va_acc = evaluate(test_loader, model)
    history.append({"epoch":ep,
                    "train_loss": tr_loss,
                    "train_acc": tr_acc,
                    "val_loss": va_loss,
                    "val_acc": va_acc})
    # save model checkpoint
    if(va_loss < best_loss):
        # update best loss
        best_loss = va_loss

        # save the parameter
        torch.save(model.state_dict(), "model_best.pth")

    print(f"[{ep:02d}] tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f} | va_loss={va_loss:.4f} va_acc={va_acc:.4f}")

# store metrics
os.makedirs("runs", exist_ok=True)
pd.DataFrame(history).to_csv("runs/metrics_point.csv", index=False)


# GPT DEBUG
# reload_model = RFRNN().to(device)
# reload_model.load_state_dict(torch.load("model_best.pth", map_location=device))
# reload_model.eval()

# va_loss2, va_acc2 = evaluate(test_loader, reload_model)
# print("Re-loaded checkpoint val_acc:", va_acc2)