import re, os, glob
import pandas as pd

DATA_DIR = "dataset"
OUT_MANIFEST = "manifest.csv"

def index_to_label(idx: int) -> str:
    # check index (even or odd)
    if idx % 2 == 1:
        return "normal"
    else:
        return "point"

def label_from_filename(path: str) -> str:
    # return the anomaly class label as string
    path_idx = int(path)

    # return label
    return index_to_label(path_idx)

# fetch through all the dataset in the directory
rows = []
for parent in os.listdir(DATA_DIR):
    for idx in glob.glob(os.path.join(DATA_DIR, parent, "*.csv")):
        # get label from directory name
        lbl = label_from_filename(parent[-2:])
        rows.append({"path": idx, "label": lbl})

pd.DataFrame(rows).to_csv(OUT_MANIFEST, index=False)
print(f"Wrote {OUT_MANIFEST} with {len(rows)} rows")