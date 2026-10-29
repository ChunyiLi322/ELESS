# coding=utf-8
# A lightweight BERT/Transformer finetuning runner using Hugging Face `transformers`.
# Drop-in replacement for classic BERT runners, but defaults to compact models.
#
# Highlights:
# - Works with DistilBERT / TinyBERT / MobileBERT / MiniLM / ALBERT via --bert_model
# - Uses modern AdamW + linear warmup scheduler
# - Supports fp16 via torch.cuda.amp (no Apex required)
# - Avoids token_type_ids for maximum model compatibility (DistilBERT doesn't use them)
# - Keeps your original processors and CLI shape as much as possible

from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import sys
import re
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup,
    BertTokenizer,  # <-- added for offline vocab.txt load
)
from transformers import AutoModel


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"



# -----------------------------
# Utils & Logging
# -----------------------------

def set_logging():
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
    )
    return logging.getLogger(__name__)

logger = set_logging()


def clean_str(string, TREC=False):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \\( ", string)
    string = re.sub(r"\)", " \\) ", string)
    string = re.sub(r"\?", " \\? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()


# -----------------------------
# Data Structures
# -----------------------------

class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class DataProcessor(object):
    def get_train_examples(self, data_dir):
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        raise NotImplementedError()

    def get_labels(self):
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)  # noqa: F821
                lines.append(line)
            return lines

    @classmethod
    def _read_corpus(cls, input_file, clean=True, MR=True, encoding='utf8', shuffle=False, lower=True):
        lines = []
        with open(input_file, encoding=encoding, errors='ignore') as fin:
            for line in fin:
                if MR:
                    label, sep, text = line.partition(' ')
                else:
                    label, sep, text = line.partition(',')
                if clean:
                    text = clean_str(text.strip()) if clean else text.strip()
                if lower:
                    text = text.lower()
                lines.append((text, label))
        if shuffle:
            perm = list(range(len(lines)))
            random.shuffle(perm)
            lines = [lines[i] for i in perm]
        return lines


# -----------------------------
# Processors (kept)
# -----------------------------

class MrpcProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        logger.info("LOOKING AT %s", os.path.join(data_dir, "train.tsv"))
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            label = line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "multinli_1.0_train.txt")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "multinli_1.0_dev_mismatched.txt")), "dev_matched")

    def get_labels(self):
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        parentheses_table = str.maketrans({'(': None, ')': None})
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[1]
            text_b = line[2]
            label = line[0]
            if label == '-':
                continue
            text_a = text_a.translate(parentheses_table)
            text_b = text_b.translate(parentheses_table)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class SnliProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "snli_1.0_train.txt")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "snli_1.0_test.txt")), "dev_matched")

    def get_labels(self):
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[5]
            text_b = line[6]
            label = line[0].lower()
            if label == '-':
                continue
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class ColaProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            label = line[1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class MRProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_corpus(os.path.join(data_dir, "train_advplus.txt")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_corpus(os.path.join(data_dir, "test.txt")), "dev")

    def get_labels(self):
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class AGProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_corpus(os.path.join(data_dir, "train_tok.csv"), MR=False, clean=False), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_corpus(os.path.join(data_dir, "test_tok.csv"), MR=False, clean=False), "dev")

    def get_labels(self):
        return ["1", "2", "3", "4"]

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class IMDBProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_corpus(os.path.join(data_dir, "train_tok.csv"), MR=True, clean=False, shuffle=True), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_corpus(os.path.join(data_dir, "test_tok.csv"), MR=True, clean=False, shuffle=True), "dev")

    def get_labels(self):
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class YelpProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_corpus(os.path.join(data_dir, "train_tok.csv"), MR=False, clean=False), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_corpus(os.path.join(data_dir, "test_tok.csv"), MR=False, clean=False), "dev")

    def get_labels(self):
        return ["1", "2"]

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class FakeProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_corpus(os.path.join(data_dir, "train_tok.csv"), MR=False, clean=False), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_corpus(os.path.join(data_dir, "test_tok.csv"), MR=False, clean=False), "dev")

    def get_labels(self):
        return ["1", "2"]

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


# -----------------------------
# Feature conversion
# -----------------------------

def _batch_encode(tokenizer, texts, max_len):
    # Works for both slow/fast tokenizers and older versions
    if hasattr(tokenizer, "__call__"):
        return tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=max_len,
            return_tensors='pt'
        )
    else:
        return tokenizer.batch_encode_plus(
            texts,
            padding='max_length',
            truncation=True,
            max_length=max_len,
            return_tensors='pt'
        )

def convert_examples_to_tensors(examples: List[InputExample], label_list: List[str], max_seq_length: int, tokenizer) -> Tuple[TensorDataset, dict]:
    label_map = {label: i for i, label in enumerate(label_list)}
    texts = []
    for ex in examples:
        if ex.text_b is not None:
            texts.append((ex.text_a, ex.text_b))
        else:
            texts.append(ex.text_a)
    enc = _batch_encode(tokenizer, texts, max_seq_length)
    labels = torch.tensor([label_map[ex.label] for ex in examples], dtype=torch.long)
    dataset = TensorDataset(enc['input_ids'], enc['attention_mask'], labels)
    return dataset, label_map


# -----------------------------
# Training / Evaluation Helpers
# -----------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def accuracy(logits: np.ndarray, labels: np.ndarray) -> float:
    preds = np.argmax(logits, axis=1)
    return float((preds == labels).sum())


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()

    # Required
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--bert_model', type=str, required=True,
                        help='HF model name OR local path (offline). e.g., /root/autodl-fs/TextRSS/models/bert_small')
    parser.add_argument('--task_name', type=str, required=True, help='cola | mnli | snli | mrpc | mr | ag | imdb | yelp | fake')
    parser.add_argument('--output_dir', type=str, required=True)

    # Other
    parser.add_argument('--cache_dir', type=str, default='')
    parser.add_argument('--max_seq_length', type=int, default=128)
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_eval', action='store_true')
    parser.add_argument('--do_resume', action='store_true')
    parser.add_argument('--do_lower_case', action='store_true')
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--eval_batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--num_train_epochs', type=float, default=5.0)
    parser.add_argument('--warmup_proportion', type=float, default=0.1)
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--fp16', action='store_true', help='Use torch.cuda.amp mixed precision')

    args = parser.parse_args()

    processors = {
        'cola': ColaProcessor,
        'mnli': MnliProcessor,
        'snli': SnliProcessor,
        'mrpc': MrpcProcessor,
        'mr': MRProcessor,
        'ag': AGProcessor,
        'imdb': IMDBProcessor,
        'yelp': YelpProcessor,
        'fake': FakeProcessor,
    }

    num_labels_task = {
        'cola': 2,
        'mnli': 3,
        'snli': 3,
        'mrpc': 2,
        'mr': 2,
        'ag': 4,
        'imdb': 2,
        'yelp': 2,
        'fake': 2,
    }

    task_name = args.task_name.lower()
    if task_name not in processors:
        raise ValueError('Task not found: %s' % task_name)

    # Device / DDP
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        n_gpu = 1
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: %s n_gpu: %s, distributed: %s", device, n_gpu, bool(args.local_rank != -1))

    if args.gradient_accumulation_steps < 1:
        raise ValueError('gradient_accumulation_steps must be >= 1')

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    set_seed(args.seed)

    if not args.do_train and not args.do_eval and not args.do_resume:
        raise ValueError('At least one of --do_train, --do_eval, --do_resume must be set.')

    os.makedirs(args.output_dir, exist_ok=True)

    processor = processors[task_name]()
    label_list = processor.get_labels()
    num_labels = num_labels_task[task_name]

    # === Offline/local model handling ===
    is_local = os.path.isdir(args.bert_model)
    
    print("---------------1---------------------")
    if is_local:
        # Optional but helpful: block any accidental internet calls
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        vocab_file = os.path.join(args.bert_model, "vocab.txt")
        # print("---------------vocab_file---------------------",os.path.exists(vocab_file))
        
        if os.path.exists(vocab_file):
            # Strict offline: load from vocab.txt directly to avoid needing tokenizer_config.json
            tokenizer = BertTokenizer(vocab_file=vocab_file, do_lower_case=args.do_lower_case)
        else:
            # Fallback to AutoTokenizer but keep local-only and disable fast
            tokenizer = AutoTokenizer.from_pretrained(
                args.bert_model, use_fast=False, do_lower_case=args.do_lower_case, local_files_only=True
            )

        config = AutoConfig.from_pretrained(args.bert_model, num_labels=num_labels, local_files_only=True)

        if args.do_resume and not args.do_train and os.path.isdir(args.output_dir):
            model = AutoModelForSequenceClassification.from_pretrained(args.output_dir, config=config, local_files_only=True)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(args.bert_model, config=config, local_files_only=True)

    else:
        # Online / cache mode
        tokenizer = AutoTokenizer.from_pretrained(args.bert_model, use_fast=True, do_lower_case=args.do_lower_case)
        config = AutoConfig.from_pretrained(args.bert_model, num_labels=num_labels)
        if args.do_resume and not args.do_train and os.path.isdir(args.output_dir):
            model = AutoModelForSequenceClassification.from_pretrained(args.output_dir, config=config)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(args.bert_model, config=config)
    print("---------------2---------------------")
    model.to(device)
    if args.local_rank == -1 and n_gpu > 1:
        model = torch.nn.DataParallel(model)
    print("---------------2、1---------------------")
    # -------- Training --------
    global_step = 0
    tr_loss = 0.0
    if args.do_train:
        print("---------------2、2---------------------")
        train_examples = processor.get_train_examples(args.data_dir)
        print("---------------2、3---------------------")
        train_dataset, _ = convert_examples_to_tensors(train_examples, label_list, args.max_seq_length, tokenizer)
        print("---------------2、4---------------------")  
        train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
        
        print("---------------2、5--------------------") 
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

        t_total = (len(train_dataloader) // args.gradient_accumulation_steps) * int(args.num_train_epochs)

        # Optimizer & Scheduler
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': 0.01,
            },
            {
                'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(args.warmup_proportion * max(1, t_total)),
            num_training_steps=max(1, t_total),
        )

        scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
        print("---------------3---------------------")

        model.train()
        for epoch in trange(int(args.num_train_epochs), desc='Epoch'):
            epoch_iterator = tqdm(train_dataloader, desc='Iteration')
            for step, batch in enumerate(epoch_iterator):
                input_ids, attention_mask, labels = (t.to(device) for t in batch)

                with torch.cuda.amp.autocast(enabled=args.fp16):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    if n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
                        loss = loss.mean()
                    loss = loss / args.gradient_accumulation_steps

                scaler.scale(loss).backward()

                tr_loss += loss.item()

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    global_step += 1

        # Save model & tokenizer
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

    # -------- Evaluation --------
    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        eval_examples = processor.get_dev_examples(args.data_dir)
        eval_dataset, _ = convert_examples_to_tensors(eval_examples, label_list, args.max_seq_length, tokenizer)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model.eval()
        eval_loss = 0.0
        eval_accuracy = 0.0
        nb_eval_steps = 0
        nb_eval_examples = 0

        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc='Evaluating'):
                input_ids, attention_mask, labels = (t.to(device) for t in batch)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                tmp_eval_loss = outputs.loss
                logits = outputs.logits

                logits_np = logits.detach().cpu().numpy()
                labels_np = labels.detach().cpu().numpy()
                tmp_eval_accuracy = accuracy(logits_np, labels_np)

                eval_loss += tmp_eval_loss.mean().item()
                eval_accuracy += tmp_eval_accuracy
                nb_eval_examples += input_ids.size(0)
                nb_eval_steps += 1

        eval_loss = eval_loss / max(1, nb_eval_steps)
        eval_accuracy = eval_accuracy / max(1, nb_eval_examples)
        result = {
            'eval_loss': eval_loss,
            'eval_accuracy': eval_accuracy,
            'global_step': global_step,
            'loss': (tr_loss / global_step) if (global_step > 0 and args.do_train) else None,
        }

        os.makedirs(args.output_dir, exist_ok=True)
        output_eval_file = os.path.join(args.output_dir, 'eval_results.txt')
        with open(output_eval_file, 'w') as writer:
            logger.info('***** Eval results *****')
            for k in sorted(result.keys()):
                logger.info('  %s = %s', k, str(result[k]))
                writer.write('%s = %s\n' % (k, str(result[k])))


if __name__ == '__main__':
    main()
