from MTGpred.model.dataset import JoinedMatchesDataset
from MTGpred.utils.mtgjson import load_cards_df
import json
from sklearn.model_selection import train_test_split

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
import numpy as np
import wandb
import pandas as pd

matches_df = pd.read_csv("data/matches.csv")
matches_df["p1_deck"] = matches_df["p1_deck"].apply(eval)
matches_df["p2_deck"] = matches_df["p2_deck"].apply(eval)

cards_df = load_cards_df(data_path="data/AtomicCards.json")

# Split the data into train and test
train_matches, test_matches = train_test_split(
    matches_df, test_size=0.2, random_state=42
)

train_dataset = JoinedMatchesDataset(
    cards_df,
    train_matches,
    model_name="allenai/longformer-base-4096",
    cased=False,
    max_length=4096,
)
test_dataset = JoinedMatchesDataset(
    cards_df,
    test_matches,
    model_name="allenai/longformer-base-4096",
    cased=False,
    max_length=4096,
)

# Cargar métricas
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")
precision = evaluate.load("precision")
recall = evaluate.load("recall")

LABELS_NAMES = ["WIN", "LOSS"]


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    # Calcular métricas
    accuracy_result = accuracy.compute(predictions=predictions, references=labels)
    f1_result = f1.compute(predictions=predictions, references=labels, average="macro")
    precision_result = precision.compute(
        predictions=predictions, references=labels, average="macro"
    )
    recall_result = recall.compute(
        predictions=predictions, references=labels, average="macro"
    )

    wandb.log(
        {
            "confusion_matrix": wandb.plot.confusion_matrix(
                probs=None, y_true=labels, preds=predictions, class_names=LABELS_NAMES
            )
        }
    )

    return {
        "accuracy": accuracy_result["accuracy"],
        "f1": f1_result["f1"],
        "precision": precision_result["precision"],
        "recall": recall_result["recall"],
    }


model = AutoModelForSequenceClassification.from_pretrained(
    "models/longformer_v3", num_labels=2, ignore_mismatched_sizes=True
)

training_args = TrainingArguments(
    output_dir="models/longformer_preds_checkpoints",
    learning_rate=1e-5,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=100,
    load_best_model_at_end=True,
    gradient_checkpointing=True,
    logging_steps=10,
    lr_scheduler_type="cosine",
    warmup_ratio=0.2,
    no_cuda=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

wandb.init(project="MTGpred", entity="javier-jimenez99")
trainer.train()
wandb.finish()
