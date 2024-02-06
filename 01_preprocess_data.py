import argparse
import collections
import glob
import gzip
from interval import Interval
import io
import json
import os
import stanza
import pickle
import pyarrow as pa
import yaml

from datasets import dataset_dict, Dataset

import iclr_lib

parser = argparse.ArgumentParser(description="")
parser.add_argument("-d", "--data_dir", default="data/", type=str, help="")

with open('schema.yml', "r") as f:
    SCHEMA = yaml.safe_load(io.StringIO(f.read()))


# ==== DISAPERE

disapere_pol_map = {"none": "non", "pol_negative": "neg", "pol_positive": "pos"}

disapere_asp_map = {
    "arg_other": "non",
    "asp_clarity": "clr",
    "asp_meaningful-comparison": "mng",
    "asp_motivation-impact": "mot",
    "asp_originality": "org",
    "asp_replicability": "rep",
    "asp_soundness-correctness": "snd",
    "asp_substance": "sbs",
    "none": "non",
}


def get_disapere_labels(sent):
    labels = {
        "pol": disapere_pol_map[sent["polarity"]],
        "asp": disapere_asp_map[sent["aspect"]],
    }
    labels["epi"] = (
        "epi" if sent["review_action"] in ["arg_request", "arg_evaluative"] else "nep"
    )
    return labels

def build_dataset(train, dev, test, task):
    label2id = {label:i for i, label in enumerate(SCHEMA['labels'][task])}
    dataset_dict_builder = {}
    for subset, lines in zip("train dev test".split(), [train, dev, test]):
        dataset_dict_builder[subset] = Dataset(
        pa.Table.from_arrays([
            [x['text'] for x in lines],
            [label2id[x['label']] for x in lines]], names = ['text', 'label']))
    return dataset_dict.DatasetDict(dataset_dict_builder), label2id



def preprocess_disapere(data_dir):

    lines_by_subset = {}
    for subset in "train dev test".split():
        lines = collections.defaultdict(list)
        for filename in glob.glob(f"{data_dir}/raw/disapere/{subset}/*.json"):
            with open(filename, "r") as f:
                obj = json.load(f)
                review_id = obj["metadata"]["review_id"]
                identifier_prefix = f"disapere|{subset}|{review_id}|"
                for sent in obj["review_sentences"]:
                    for task, label in get_disapere_labels(sent).items():
                        lines[task].append(
                            {
                                "identifier": f'{identifier_prefix}{sent["sentence_index"]}',
                                "text": sent["text"],
                                "label": label,
                            }
                        )
        lines_by_subset[subset] = lines

    for task in lines_by_subset["train"]:
        dataset = build_dataset(
            lines_by_subset['train'][task],
            lines_by_subset['dev'][task],
            lines_by_subset['test'][task],
            task
        )
        output_dir = f"{data_dir}/labeled/{task}/"
        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/disapere.pkl", "wb") as f:
            pickle.dump(dataset, f)


# ==== AMPERE

ampere_epi_map = {
    "non-arg": "nep",
    "evaluation": "epi",
    "request": "epi",
    "fact": "nep",
    "reference": "nep",
    "quote": "nep",
}


def preprocess_ampere(data_dir):

    examples = []

    for filename in glob.glob(f"{data_dir}/raw/ampere/*.txt"):
        review_id = filename.split("/")[-1].rsplit(".", 1)[0].split("_")[0]
        with open(filename, "r") as f:
            sentence_dicts = []
            for i, line in enumerate(f):
                label, sentence = line.strip().split("\t", 1)
                examples.append(
                    {
                        "identifier": f"ampere|train|{review_id}|{i}",
                        "text": sentence,
                        "label": ampere_epi_map[label],
                    }
                )
    with open(f"{data_dir}/labeled/epi/train/ampere.jsonl", "w") as f:
        f.write("\n".join(json.dumps(e) for e in examples))


# ==== ReviewAdvisor

SENTENCIZE_PIPELINE = stanza.Pipeline("en", processors="tokenize")
TOLERANCE = 7

revadv_label_map = {
    "positive": "pos",
    "negative": "neg",
    "clarity": "clr",
    "meaningful_comparison": "mng",
    "motivation": "mot",
    "originality": "org",
    "replicability": "rep",
    "soundness": "snd",
    "substance": "sbs",
}

Sentence = collections.namedtuple("Sentence", "interval text")


def tokenize(text):
    doc = SENTENCIZE_PIPELINE(text)
    sentences = []
    for sentence in doc.sentences:
        start = sentence.to_dict()[0]["start_char"]
        end = sentence.to_dict()[-1]["end_char"]
        sentences.append(Sentence(Interval(start, end), sentence.text))
    return sentences


def label_sentences(sentences, label_obj):
    labels = [list() for _ in range(len(sentences))]
    for label_start, label_end, label in label_obj:
        label_interval = Interval(label_start, label_end)
        for i, sentence in enumerate(sentences):
            if label_interval == sentence.interval:
                labels[i].append(label)
            elif (
                label_start > sentence.interval.upper_bound
                or label_end < sentence.interval.lower_bound
            ):
                pass
            else:
                overlap = sentence.interval & label_interval
                if overlap.upper_bound - overlap.lower_bound > TOLERANCE:
                    labels[i].append(label)
    return labels


def preprocess_revadv(data_dir):

    with gzip.open(f"{data_dir}/raw/revadv/review_with_aspect.jsonl.gz", "r") as f:
        lines = collections.defaultdict(list)
        for line in f:
            obj = json.loads(line)
            identifier_prefix = f'revadv|train|{obj["id"]}|'
            sentences = tokenize(obj["text"])
            revadv_labels = label_sentences(sentences, obj["labels"])
            converted_labels = {}
            for i, (sentence, label_list) in enumerate(zip(sentences, revadv_labels)):
                if not label_list or "summary" in label_list:
                    converted_labels = {"epi": "nep", "pol": "non", "asp": "non"}
                else:
                    converted_labels["epi"] = "epi"
                    asp, pol = label_list[0].rsplit("_", 1)
                    converted_labels = {
                        "epi": "epi",
                        "pol": revadv_label_map[pol],
                        "asp": revadv_label_map[asp],
                    }
                    for task, label in converted_labels.items():
                        lines[task].append(
                            {
                                "identifier": f"{identifier_prefix}{i}",
                                "text": sentence.text,
                                "label": label,
                            }
                        )
    for task, examples in lines.items():
        with open(f"{data_dir}/labeled/{task}/train/revadv.jsonl", "w") as f:
            f.write("\n".join(json.dumps(e) for e in examples))


def prepare_unlabeled_iclr_data(data_dir):
    lines = collections.defaultdict(list)
    for filename in glob.glob(f"{data_dir}/raw/iclr/*.json"):
        with open(filename, "r") as f:
            obj = json.load(f)
            review_id = obj["identifier"]
            identifier_prefix = f"iclr|predict|{review_id}|"
            for i, sent in enumerate(tokenize(obj["text"])):
                for task in "epi pol asp".split():
                    lines[task].append(
                        {
                            "identifier": f"{identifier_prefix}{i}",
                            "text": sent.text,
                            "label": None,
                        }
                    )
    for task, examples in lines.items():
        output_dir = f"{data_dir}/unlabeled/{task}/predict/"
        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/iclr.jsonl", "w") as f:
            f.write("\n".join(json.dumps(e) for e in examples))


def main():

    args = parser.parse_args()

    print("Preprocessing DISAPERE")
    preprocess_disapere(args.data_dir)
    print("Preprocessing AMPERE")
    preprocess_ampere(args.data_dir)
    print("Preprocessing ReviewAdvisor")
    preprocess_revadv(args.data_dir)
    #print("Downloading ICLR data")
    #iclr_lib.get_iclr_data(f'{args.data_dir}/raw/iclr/')
    #print("Preprocessing unlabeled ICLR data")
    #prepare_unlabeled_iclr_data(args.data_dir)


if __name__ == "__main__":
    main()
