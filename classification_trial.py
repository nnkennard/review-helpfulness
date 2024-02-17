import collections
from datasets import load_dataset, dataset_dict, Dataset
import evaluate
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import pickle
import pyarrow as pa
import torch

from peft import get_peft_model, LoraConfig, TaskType

#MyDataset = collections.namedtuple("MyDataset",
#                                   "name examples label2id".split())

def preprocess_function(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True)


def train_and_eval_distillbert(dataset, tokenizer):

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    num_labels = len(dataset.label2id)
    id2label = {i: l for l, i in dataset.label2id.items()}
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


def compute_metrics(eval_pred):
    # All metrics are already predefined in the HF `evaluate` package
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")
    accuracy_metric = evaluate.load("accuracy")

    logits, labels = eval_pred  # eval_pred is the tuple of predictions and labels returned by the model
    predictions = np.argmax(logits, axis=-1)
    #precision = precision_metric.compute(predictions=predictions,
    #                                     references=labels)["precision"]
    #recall = recall_metric.compute(predictions=predictions,
    #                               references=labels)["recall"]
    #f1 = f1_metric.compute(predictions=predictions, references=labels)["f1"]
    accuracy = accuracy_metric.compute(predictions=predictions,
                                       references=labels)["accuracy"]
    # The trainer is expecting a dictionary where the keys are the metrics names and the values are the scores.
    return {
     #   "precision": precision,
      #  "recall": recall,
       # "f1-score": f1,
        'accuracy': accuracy
    }


def llama_preprocessing_function(examples, llama_tokenizer):
    MAX_LEN = 500
    return llama_tokenizer(examples['text'],
                           truncation=True,
                           max_length=MAX_LEN)


class WeightedCELossTrainer(Trainer):

    def __init__(self, model, args, train_dataset, eval_dataset, data_collator,
    compute_metrics, weights):
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics
            )
        self.weights = weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        # Get model's predictions
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # Compute custom loss
        loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor(
            self.weights, device=model.device, dtype=logits.dtype))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels),
                        labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def calculate_weights(train_dataset):
    weight_counts = collections.Counter(x['label'] for x in train_dataset)
    total_samples = len(train_dataset)
    num_classes = len(weight_counts)
    print(weight_counts.keys())
    assert set(weight_counts.keys()) == set(range(len(weight_counts)))
    return [total_samples/(num_classes * weight_counts[i]) for i in
    range(num_classes)]

def llama(dataset, label2id):
    llama_checkpoint = "/datasets/ai/llama2/huggingface/llama-2-7b"
    llama_tokenizer = AutoTokenizer.from_pretrained(llama_checkpoint,
                                                    add_prefix_space=True)
    llama_tokenizer.pad_token_id = llama_tokenizer.eos_token_id
    llama_tokenizer.pad_token = llama_tokenizer.eos_token

    weights = calculate_weights(dataset['train'])
    llama_tokenized_datasets = dataset.map(
        lambda x: llama_preprocessing_function(x, llama_tokenizer),
        batched=True)
    llama_tokenized_datasets.set_format("torch")
    llama_data_collator = DataCollatorWithPadding(tokenizer=llama_tokenizer)

    llama_model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=llama_checkpoint,
        num_labels=len(weights),
        device_map="auto",
        offload_folder="offload",
        trust_remote_code=True)

    llama_model.config.pad_token_id = llama_model.config.eos_token_id

    llama_peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj",
            "v_proj",
        ],
    )

    llama_model = get_peft_model(llama_model, llama_peft_config)
    llama_model.print_trainable_parameters()

    llama_model = llama_model.cuda()

    lr = 1e-4
    batch_size = 8
    num_epochs = 5
    training_args = TrainingArguments(
        output_dir="llama-lora-token-classification",
        learning_rate=lr,
        lr_scheduler_type="constant",
        warmup_ratio=0.1,
        max_grad_norm=0.3,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.001,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        #report_to="wandb",
        fp16=True,
        gradient_checkpointing=True,
    )

    llama_trainer = WeightedCELossTrainer(
    #llama_trainer = Trainer(
        model=llama_model,
        args=training_args,
        train_dataset=llama_tokenized_datasets['train'],
        eval_dataset=llama_tokenized_datasets["dev"],
        data_collator=llama_data_collator,
        compute_metrics=compute_metrics,
        weights=weights,
    )

    llama_trainer.train()

def mixtral():
    pass


def main():

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    with open('data/labeled/pol.pkl', 'rb') as f:
        untokenized, label2id = pickle.load(f)

    llama(untokenized, label2id)

    exit()

    #max_char = untokenized['train'].to_pandas()['text'].str.len().max()
    # Number of Words
    #max_words = untokenized['train'].to_pandas()['text'].str.split().str.len(
    #).max()

    #print(max_char, max_words)
    #dsds

    # Specific to imdb dataset
    #tokenized = untokenized.map(lambda x: preprocess_function(x, tokenizer),
                                batched=True)
    #dataset = MyDataset("asp", tokenized, label2id)

    #train_and_eval_distillbert(dataset, tokenizer)


if __name__ == "__main__":
    main()
