"""Fine-tune ParsBERT, XLM-R and mBERT (5 seeds x 2 tasks) and save the test predictions.

Usage: python 02_finetune_baselines.py [--models parsbert xlmr mbert]
A GPU is recommended. Finished runs are skipped, so the script can be restarted.
"""
import argparse
import gc
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          Trainer, TrainingArguments, set_seed)

from common import PRED_DIR, RESULTS, SEEDS, SPLITS, TASKS, read_csv

CHECKPOINTS = {
    "parsbert": "HooshvareLab/bert-fa-base-uncased",
    "xlmr": "xlm-roberta-base",
    "mbert": "bert-base-multilingual-cased",
}
EPOCHS, BATCH_SIZE, LEARNING_RATE, MAX_LENGTH = 4, 16, 2e-5, 128
LOG = RESULTS / "all_runs_log.csv"
transformers.logging.set_verbosity_error()


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, label2id):
        self.encodings = tokenizer(df["text"].tolist(), truncation=True, max_length=MAX_LENGTH)
        self.labels = df["label"].map(label2id).tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {key: values[i] for key, values in self.encodings.items()}
        item["labels"] = self.labels[i]
        return item


class WeightedTrainer(Trainer):
    """Trainer with class-weighted cross-entropy loss."""

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = torch.nn.functional.cross_entropy(
            outputs.logits, labels, weight=self.class_weights.to(outputs.logits.device))
        return (loss, outputs) if return_outputs else loss


def compute_metrics(prediction):
    predicted = prediction.predictions.argmax(-1)
    return {"accuracy": accuracy_score(prediction.label_ids, predicted),
            "f1": f1_score(prediction.label_ids, predicted, average="macro")}


def finished_runs():
    return set(pd.read_csv(LOG)["run"]) if LOG.exists() else set()


def run_one(key, task, seed):
    run = f"{key}_{task}_s{seed}"
    if run in finished_runs():
        print("skip:", run)
        return
    print("start:", run, time.strftime("%H:%M:%S"))

    train, val, test = (read_csv(SPLITS / f"{task}_{name}.csv") for name in ["train", "val", "test"])
    labels = sorted(train["label"].unique())
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINTS[key])
    train_ds, val_ds, test_ds = (TextDataset(df, tokenizer, label2id) for df in (train, val, test))
    class_weights = torch.tensor(
        compute_class_weight("balanced", classes=np.arange(len(labels)), y=np.array(train_ds.labels)),
        dtype=torch.float)

    set_seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(
        CHECKPOINTS[key], num_labels=len(labels), id2label=id2label, label2id=label2id)
    warmup_steps = int(0.1 * EPOCHS * -(-len(train_ds) // BATCH_SIZE))

    with tempfile.TemporaryDirectory() as tmp:
        args = TrainingArguments(
            output_dir=tmp, num_train_epochs=EPOCHS, per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=32, learning_rate=LEARNING_RATE, weight_decay=0.01,
            warmup_steps=warmup_steps, eval_strategy="epoch", save_strategy="epoch", save_total_limit=2,
            load_best_model_at_end=True, metric_for_best_model="f1", seed=seed,
            fp16=torch.cuda.is_available(), report_to="none", disable_tqdm=True)
        trainer = WeightedTrainer(
            model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
            data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=compute_metrics,
            class_weights=class_weights)

        start = time.time()
        trainer.train()
        train_minutes = round((time.time() - start) / 60, 1)

        best_checkpoint = trainer.state.best_model_checkpoint
        best_step = int(Path(best_checkpoint).name.split("-")[-1])
        best_model = AutoModelForSequenceClassification.from_pretrained(best_checkpoint)
        predictor = Trainer(
            model=best_model,
            args=TrainingArguments(output_dir=tmp, per_device_eval_batch_size=32,
                                   fp16=torch.cuda.is_available(), report_to="none", disable_tqdm=True),
            data_collator=DataCollatorWithPadding(tokenizer))
        start = time.time()
        output = predictor.predict(test_ds)
        inference_seconds = round(time.time() - start, 1)
        val_f1 = trainer.state.best_metric

    predicted = output.predictions.argmax(-1)
    true = np.array(test_ds.labels)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"text": test["text"], "true": [id2label[i] for i in true],
                  "pred": [id2label[i] for i in predicted]}
                 ).to_csv(PRED_DIR / f"preds_{run}.csv", index=False, encoding="utf-8-sig")

    row = pd.DataFrame([{
        "run": run, "model": key, "task": task, "seed": seed, "best_step": best_step,
        "val_f1": round(val_f1, 4), "test_acc": round(accuracy_score(true, predicted), 4),
        "test_f1": round(f1_score(true, predicted, average="macro"), 4),
        "train_min": train_minutes, "infer_sec": inference_seconds}])
    row.to_csv(LOG, mode="a", header=not LOG.exists(), index=False)
    print(f"done: val F1 {val_f1:.4f} | test accuracy {row['test_acc'][0]:.4f} | {train_minutes} min")

    del trainer, predictor, model, best_model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(CHECKPOINTS), choices=list(CHECKPOINTS))
    selected = parser.parse_args().models
    RESULTS.mkdir(parents=True, exist_ok=True)
    for model_key in selected:
        for task_name in TASKS:
            for seed_value in SEEDS:
                run_one(model_key, task_name, seed_value)
