import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

# data loader class
class CSVSequenceDataset(Dataset):
    def __init__(self, manifest_csv, label_map=None, seq_len=500):
        self.df = pd.read_csv(manifest_csv)
        self.seq_len = seq_len
        if label_map is None:
            classes = sorted(self.df["label"].unique().tolist())
            self.label_map = {c:i for i, c in enumerate(classes)}
        else:
            self.label_map = label_map

    def __len__(self):
        return len(self.df)

    def _load_series(self, path):
        raw = pd.read_csv(path)
        s = raw.iloc[:, 0].astype(float).values
        return torch.tensor(s, dtype=torch.float32).unsqueeze(-1)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        series = self._load_series(row["path"])

            # Convert to torch tensor
        x = torch.as_tensor(series, dtype=torch.float32)

        # Ensure shape is (1, L) for Conv1d
        if x.dim() == 1:
            # (L,) -> (1, L)
            x = x.unsqueeze(0)
        elif x.dim() == 2:
            # Could be (L, 1) or (1, L); we want (1, L)
            if x.shape[0] != 1 and x.shape[1] == 1:
                # (L,1) -> (1,L)
                x = x.transpose(0, 1)
        # if x.shape == (1,L) already, we leave it

        y = self.label_map[row["label"]]
        return x, torch.tensor(y, dtype=torch.long)