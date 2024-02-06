import collections
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
import numpy as np


MyDataset = collections.namedtuple("MyDataset", 
    "name examples id2label".split())



ACCURACY = evaluate.load('accuracy')


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return ACCURACY.compute(predictions=predictions, references=labels)


def preprocess_function(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True)

def train_and_eval_distillbert(dataset, tokenizer):

    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


    num_labels = len(dataset.id2label)
    label2id = {l:i for i, l in dataset.id2label.items()}
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=num_labels,
        id2label=dataset.id2label,
        label2id=label2id)

    run_name = f'{dataset.name}_distillbert'
    training_args = TrainingArguments(
        output_dir=run_name,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
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
        eval_dataset=dataset.examples["test"],
        tokenizer=tokenizer, # Why is the tokenizer needed?
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()



def main():

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


    # Specific to imdb dataset
    imdb = load_dataset('imdb')
    tokenized_imdb = imdb.map(lambda x: preprocess_function(x, tokenizer),
                              batched=True)
    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    imdb_dataset = MyDataset("imdb", tokenized_imdb, id2label)

    train_and_eval_distillbert(imdb_dataset, tokenizer)

if __name__ == "__main__":
    main()
