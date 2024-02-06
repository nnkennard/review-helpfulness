import collections
from datasets import load_dataset, dataset_dict, Dataset
import evaluate
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import pickle
import pyarrow as pa

MyDataset = collections.namedtuple("MyDataset",
                                   "name examples label2id".split())

ACCURACY = evaluate.load('accuracy')


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return ACCURACY.compute(predictions=predictions, references=labels)


def preprocess_function(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True)


def train_and_eval_distillbert(dataset, tokenizer):

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    num_labels = len(dataset.label2id)
    id2label = {i:l for l, i in dataset.label2id.items()}
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=num_labels,
        id2label=id2label,
        label2id=dataset.label2id)

    run_name = f'{dataset.name}_distillbert'
    training_args = TrainingArguments(
        output_dir=run_name,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=10,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset.examples["train"],
        eval_dataset=dataset.examples["dev"],
        tokenizer=tokenizer,  # Why is the tokenizer needed?
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()


def main():

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    with open('data/labeled/asp/disapere.pkl', 'rb') as f:
        imdb, label2id = pickle.load(f)

    # Specific to imdb dataset
    tokenized_imdb = imdb.map(lambda x: preprocess_function(x, tokenizer),
                              batched=True)
    imdb_dataset = MyDataset("imdb", tokenized_imdb, label2id)

    train_and_eval_distillbert(imdb_dataset, tokenizer)


if __name__ == "__main__":
    main()
