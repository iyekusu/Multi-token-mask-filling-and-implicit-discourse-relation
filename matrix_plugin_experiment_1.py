import argparse
import time
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import csv
import json

from dataclasses import dataclass, field, asdict
from pytorch_lightning.callbacks import ModelSummary
from pytorch_lightning.loggers import TensorBoardLogger
from torch import no_grad
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM, Trainer, BertTokenizerFast, TrainingArguments, BertTokenizer, BertForMaskedLM, \
    HfArgumentParser, RobertaForMaskedLM, BertForSequenceClassification
import datasets
import os
from tqdm import tqdm
import pandas as pd
import torch
import pytorch_lightning as pl
import torch
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from typing import Dict

from configuration import DATA_PATH, HOME_DIR
from sense_mapping import sense_dict
import torch.nn.functional as F
import numpy as np


def create_mapping(dataset, model_name, dataset_path, model_path, dataset_suffix):
    tok: BertTokenizerFast = AutoTokenizer.from_pretrained(model_name)
    seen_dataset = datasets.DatasetDict({k: v for k, v in dataset.items() if k in ['train', 'dev', 'test_idrr', 'test_preposed', 'test_canonical'] and k in dataset})
    with open('Multi-token_list.txt', 'r') as f:
        vocab = set(f.read().splitlines())
        print(f'added tokens: {vocab}')
    print(f'Vocab before mapping: {len(tok.vocab)}')
    mapping = tok.vocab
    cur_id = len(mapping)
    for v in tqdm(vocab):
        if v not in mapping:
            mapping[v] = cur_id
            cur_id += 1

    # def tokenize_span(span):
    #     tokens = tok.tokenize(span)
    #     token_ids = [mapping[token] for token in token
    #     return token_ids   
    # seen_dataset = seen_dataset.map(lambda samples: {'labels': [tokenize_span(v) for v in samples['span']]}, batched=True, num_proc=16)
    seen_dataset = seen_dataset.map(lambda samples: {'labels': [mapping[v] for v in samples['span']]}, batched=True, num_proc=16)
    seen_dataset.save_to_disk(dataset_path)

    # Load the model and adjust embeddings
    model: BertForMaskedLM = AutoModelForMaskedLM.from_pretrained(model_name)
    print(f'Extended vocab: {len(mapping)}')
    model.resize_token_embeddings(len(mapping))
    model.save_pretrained(model_path)

    pd.Series(mapping).to_csv(f'matrix_plugin/ext_dec_map_{model_name.replace("/", "_")}{dataset_suffix}.csv', index_label='phrase', header=['id'])
    return model, seen_dataset, mapping

class MatrixDecoder(pl.LightningModule):
    def __init__(self, config, model):
        super().__init__()

        self.save_hyperparameters(config)
        self.config = config
        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, input_features):
        return self.model(input_features)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config['lr'])

    def step(self, batch):
        y = self.forward(batch['input_features'])
        loss = self.criterion(y, batch['labels'])
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log('val_loss', loss, prog_bar=True)
        return loss


def train(config: Dict, transformer_matrix: BertOnlyMLMHead, dataset: datasets.DatasetDict):
    trainer = pl.Trainer(
        devices=config['num_gpus']
        , accelerator='gpu'
        , logger=TensorBoardLogger(save_dir=os.getcwd(), version=config['version'], name='matrix_plugin_results')
        , max_epochs=config['epochs']
        , val_check_interval=config['val_check_interval']
        , gradient_clip_val=1
    )

    model = MatrixDecoder(config, transformer_matrix)
    dataset.set_format(type='pt', columns=['input_features', 'labels'])
    trainer.fit(model,
                DataLoader(dataset['train'], config['batch_size'], num_workers=16, shuffle=True),
                DataLoader(dataset['dev'].select(range(config['dev_size'])), 32, num_workers=16))

    return trainer.checkpoint_callback.best_model_path


class Logger:
    def __init__(self, logfile):
        print("output to:", logfile)
        self.logfile = logfile
        self.fp = open(logfile, 'w')

    def print(self, output):
        print(output)
        self.fp.write(str(output) + '\n')

    def close(self):
        print("output printed to:", self.logfile)
        self.fp.close()

@torch.no_grad()
def test(matrix, dataset: datasets.DatasetDict, mapping: Dict, tokenizer: BertTokenizerFast, ckpt_path, sense_dict,log=False):
    test_model = MatrixDecoder.load_from_checkpoint(ckpt_path, model=matrix).cuda().eval()
    test_dataset = dataset['test_canonical']
    rev_map = {v: k for k, v in mapping.items()}
    acc_at_to_check = [1, 2, 3, 4, 5, 10, 20, 50, 100]
    found, total = defaultdict(int), 0
    PRINT_EVERY = 20
    surprisals, entropies = [], []
    total_top_n = defaultdict(int)
    correct_top_n = defaultdict(int)

    logger = Logger(Path(ckpt_path).parent.parent / 'top5_res_canonical_epoch4.log')
    if log:
        top5_fp = open(Path(ckpt_path).parent.parent / 'top5_res_canonical_epoch4.csv', 'w')
        writer = csv.DictWriter(top5_fp, fieldnames=[
                                                    'surprisal', 'entropy',
                                                    'top1_match', 'top5_match', 
                                                    'corpus', 
                                                    'datasource', 
                                                    'genre', 
                                                    'sense', 'connective', 'text', 'masked_text',
                                                    'preposed_phrase',
                                                    'top50_results'
                                                    ])
        writer.writeheader()

    BATCH_SZ = 128
    loader = DataLoader(test_dataset, batch_size=BATCH_SZ, shuffle=False) 
    example_list = [] 
    for irow, row in enumerate(tqdm(loader)):
        # metadata
        genre = row['genre']
        corpus = row['corpus']
        gold_senses = row['sense'] 
        texts = row['text']
        masked_texts = row['masked_text']
        connectives = row['connective']
        datasource = row['datasource'] 
        preposed_phrase = row['preposed_phrase']
        # predictions
        input_features = torch.stack(row['input_features']).type(torch.FloatTensor).T.cuda()
        logits = test_model.forward(input_features)
        results = torch.topk(logits, k=max(acc_at_to_check), dim=-1)
        text_res = [[rev_map[int(i)] for i in res] for res in results.indices]
        
        total += len(row['masked_text'])

        # Process each sample in the batch
        for i in range(len(masked_texts)):
            gold_sense = gold_senses[i]
            gold_connective = connectives[i]

            # Calculate probabilities
            probs = F.softmax(logits[i], dim=-1).cpu().numpy()
            
            # Calculate surprisal
            nll = -np.log(np.sum([probs[j] for j in range(len(probs)) if rev_map[j] in sense_dict and gold_sense in sense_dict[rev_map[j]]]))
            if not np.isinf(nll):
                surprisals.append(nll)
            else:
                print(f"corpus: {corpus[i]}\ndatasource: {datasource[i]}\ngold_sense: {gold_sense}\nconnective: {gold_connective}\nmasked_text: {masked_texts[i]}preposed_phrase: {preposed_phrase[i]}")
            
            # Calculate entropy
            entropy = -np.sum([probs[j] * np.log(probs[j]) for j in range(len(probs))])
            entropies.append(entropy)
        
            for acc_at in acc_at_to_check:
                predicted_connectives = text_res[i][:acc_at]
                correct_num = 0

                for predicted in predicted_connectives:
                    if gold_sense in sense_dict.get(predicted, []):
                        correct_num += 1
                correct_top_n[acc_at] += correct_num
                total_top_n[acc_at] += acc_at

                if correct_num > 0 :
                    found[acc_at] += 1

            # top1/top5 match
            top_1_predictions = text_res[i][:1]
            top_5_predictions = text_res[i][:5]

            top1_match = 'V' if any(gold_sense in sense_dict.get(pred, []) for pred in top_1_predictions) else 'X'
            top5_match = 'V' if any(gold_sense in sense_dict.get(pred, []) for pred in top_5_predictions) else 'X'

            if log:
                writer.writerow({
                    'entropy': entropy,
                    'surprisal': nll,
                    'top1_match': top1_match,
                    'top5_match': top5_match,
                    'corpus': corpus[i],
                    'datasource': datasource[i],
                    'genre': genre[i],
                    'sense': gold_senses[i],
                    'connective': connectives[i],
                    'text': texts[i],
                    'masked_text': masked_texts[i],
                    'preposed_phrase': preposed_phrase[i],
                    'top50_results': text_res[i][:50],
                })

    # Final log output after processing all batches
    for acc_at in acc_at_to_check:
        logger.print(f"Final accuracy at {acc_at}: {found[acc_at] / total:.2%} ({found[acc_at]} out of {total})")
        precision = correct_top_n[acc_at] / total_top_n[acc_at]
        logger.print(f"Precision at {acc_at}: {precision:.2%} ({correct_top_n[acc_at]} out of {total_top_n[acc_at]})\n")

    surprisal = np.sum(surprisals)
    avg_entropy = np.mean(entropies)

    logger.print(f"Surprisal: {surprisal}")
    logger.print(f"Average surprisal: {np.mean(surprisals)}")
    logger.print(f"Average entropy: {avg_entropy}")


    if log:
        top5_fp.close()
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='bert-base-cased')
    parser.add_argument('--version', type=str, default='version_1')
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--dataset_name', type=str, default='wiki')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--val_check_interval', type=float, default=0.25)
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--dev_size', type=int, default=20_000)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--no_log', action='store_true')
    parser.add_argument('--benchmark', action='store_true')
    args = parser.parse_args()
    model_name = args.model
    args.log = not args.no_log
    config = vars(args)

    dataset_name = args.dataset_name
    dataset_suffix = '' if dataset_name == 'wiki' else f'_{dataset_name}'

    model_path = f'matrix_plugin/ext_dec_{model_name}{dataset_suffix}'
    dataset_path = f'data/matrix_plugin/seen_dataset_{model_name}{dataset_suffix}/'

    if args.force or not os.path.exists(model_path) or not os.path.exists(dataset_path):
        if args.input_path is not None:
            input_path = args.input_path
        elif 'roberta' in model_name:
            input_path = f'{HOME_DIR}/MultiTokenCompletionData/input_data_{model_name}{dataset_suffix}'
        elif 'spanbert' in model_name:
            input_path = f'input_data_{model_name}'
        else:
            input_path = f'{HOME_DIR}/MultiTokenCompletionData/input_data_cased'

        dataset = datasets.load_from_disk(input_path)

        # Save dataset back if it was modified
        train_split_path = os.path.join(input_path, 'train')
        if 'train' in dataset and os.path.exists(train_split_path):
            dataset['train'] = datasets.load_from_disk(train_split_path)

        train_split_path = os.path.join(input_path, 'dev')
        if 'dev' in dataset and os.path.exists(train_split_path):
            dataset['dev'] = datasets.load_from_disk(train_split_path)

        test_split_path = os.path.join(input_path, 'test_preposed')
        if 'test_preposed' in dataset and os.path.exists(test_split_path):
            dataset['test_preposed'] = datasets.load_from_disk(test_split_path)
        
        test_split_path = os.path.join(input_path, 'test_canonical')
        if 'test_canonical' in dataset and os.path.exists(test_split_path):
            dataset['test_canonical'] = datasets.load_from_disk(test_split_path)

        test_split_path = os.path.join(input_path, 'test_idrr')
        if 'test_idrr' in dataset and os.path.exists(test_split_path):
            dataset['test_idrr'] = datasets.load_from_disk(test_split_path)

        print('increasing vocab')
        model, seen_dataset, mapping = create_mapping(dataset, model_name, dataset_path, model_path, dataset_suffix)
        seen_dataset.save_to_disk(dataset_path)
    else:
        print('loading vocab')
        model = AutoModelForMaskedLM.from_pretrained(model_name)
        seen_dataset = datasets.load_from_disk(dataset_path)
        mapping = pd.read_csv(f'matrix_plugin/ext_dec_map_{model_name.replace("/", "_")}{dataset_suffix}.csv', na_filter=False).set_index('phrase')[
            'id'].to_dict()
        model: BertForMaskedLM = AutoModelForMaskedLM.from_pretrained(model_path)

    if 'roberta' in model_name:
        # RobertaForMaskedLM
        matrix = model.lm_head
    else:
        # BertForMaskedLM
        matrix = model.cls

    tok = AutoTokenizer.from_pretrained(model_name)

    if args.test:
        matrix.eval()
        ckpt = args.ckpt
        if ckpt is None:
            from glob import glob
            ckpt = sorted(glob(f'matrix_plugin_results/{args.version}/checkpoints/*.ckpt'))[-1]
        print(ckpt)
        sense_dict = sense_dict
        print(sense_dict)
        test(matrix, seen_dataset, mapping, tok, ckpt, sense_dict, args.log)
    else:
        # detach output embedding matrix so it will train
        model.get_output_embeddings().weight = torch.nn.Parameter(model.get_output_embeddings().weight.clone())
        best_ckpt = train(config, matrix, seen_dataset)
        matrix.eval()
        sense_dict = sense_dict
        test(matrix, seen_dataset, mapping, tok, best_ckpt, sense_dict, args.log)
