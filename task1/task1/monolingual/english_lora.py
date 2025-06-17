import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.chdir("../..")

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
import evaluate
import numpy as np
from task1.config import ProjectPaths
import pandas as pd
import torch

paths = ProjectPaths()

# Set device
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Load and preprocess data
def load_dataset(path):
    df = pd.read_csv(path, sep='\t')
    df = df[df['label'].isin(['SUBJ', 'OBJ'])].copy()
    df['labels'] = df['label'].map({'OBJ': 0, 'SUBJ': 1})
    df = df[['sentence', 'labels']]
    return Dataset.from_pandas(df)

train_dataset = load_dataset(paths.english_data_dir / "train_en.tsv")
val_dataset = load_dataset(paths.english_data_dir / "dev_en.tsv")
test_dataset = load_dataset(paths.english_data_dir / "dev_test_en.tsv")

# Tokenization
model_name = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_fn(examples):
    return tokenizer(
        examples["sentence"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

train_dataset = train_dataset.map(tokenize_fn, batched=True)
val_dataset = val_dataset.map(tokenize_fn, batched=True)
test_dataset = test_dataset.map(tokenize_fn, batched=True)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Load model and add LoRA
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    task_type=TaskType.SEQ_CLS,
    target_modules=["query", "key", "value"]  # Adjust target modules for XLM-RoBERTa
)

model = get_peft_model(model, lora_config).to(device)

# Define metrics
f1 = evaluate.load("f1")
accuracy = evaluate.load("accuracy")
recall = evaluate.load("recall")
precision = evaluate.load("precision")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
        "f1_macro": f1.compute(predictions=preds, references=labels, average="macro")["f1"],
        "recall": recall.compute(predictions=preds, references=labels)["recall"],
        "precision": precision.compute(predictions=preds, references=labels)["precision"]
    }

# TrainingArguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=15,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# Train
trainer.train()

print("Training complete")

# Evaluate on test set
print("Evaluating on test set")
test_results = trainer.evaluate(eval_dataset=test_dataset)
print(test_results)
