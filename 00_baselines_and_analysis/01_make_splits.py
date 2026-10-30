"""Create the train/validation/test splits.

Test: the balanced evaluation sample given to the LLMs (data/input/<task>_balanced.csv).
Train/validation: the remaining cleaned texts (data/input/<task>_full.csv), split 90/10
with stratification and seed 42.
"""
from sklearn.model_selection import train_test_split

from common import INPUT, SPLITS, TASKS, read_csv

SEED = 42
SPLITS.mkdir(parents=True, exist_ok=True)

for task in TASKS:
    full = read_csv(INPUT / f"{task}_full.csv").rename(columns={task: "label"})[["text", "label"]]
    test = read_csv(INPUT / f"{task}_balanced.csv").rename(columns={task: "label"})[["text", "label"]]
    full["text"] = full["text"].astype(str).str.strip()
    test["text"] = test["text"].astype(str).str.strip()

    pool = full[~full["text"].isin(set(test["text"]))].reset_index(drop=True)
    train, val = train_test_split(pool, test_size=0.1, stratify=pool["label"], random_state=SEED)
    assert not set(test["text"]) & (set(train["text"]) | set(val["text"])), "test texts leaked into training data"

    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.to_csv(SPLITS / f"{task}_{name}.csv", index=False, encoding="utf-8-sig")
    print(f"{task}: train {len(train)} | validation {len(val)} | test {len(test)}")
