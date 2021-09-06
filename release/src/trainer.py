from collections import defaultdict
import time
import json
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from math import ceil
import torch
from torch import nn
from torch.utils.data import Dataset, TensorDataset, DataLoader, SequentialSampler
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
import numpy as np
import os
import shutil
import sys
from tqdm import tqdm
from model import PICOSentClassModel
import warnings
warnings.filterwarnings("ignore")


def f1_score(ground, preds):
    tp, tn, fp, fn = 0, 0, 0, 0
    for g, p in zip(ground, preds):
        if g == p:
            if g == 1:
                tp += 1
            else:
                tn += 1
        else:
            if g == 1:
                fn += 1
            else:
                fp += 1
    precision = tp/(tp+fp) if tp+fp > 0 else 0.
    recall = tp/(tp+fn) if tp+fp > 0 else 0.
    f1 = 2*precision*recall/(precision + recall) if precision+recall > 0 else 0.
    print((tp, tn, fp, fn), (precision, recall, f1))
    return (tp, tn, fp, fn), (precision, recall, f1)


class PICOSentClassTrainer(object):

    def __init__(self, args):
        self.args = args
        self.corpus_path = args.corpus_path
        self.train_file = args.train_file
        self.dev_file = args.dev_file
        self.test_file = args.test_file
        self.output_path = os.path.join(args.output_path, args.label_name, '_pool_', args.eval_metric)
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        self.label_name = args.label_name
        self.label_query_name = {'aggregation': 'agg_labels',
                                 'major': 'major_labels',
                                 'minor': 'minor_labels'}[self.label_name]

        self.eval_batch_size = args.eval_batch_size
        self.train_batch_size = args.train_batch_size
        self.max_len = args.max_len
        self.accum_steps = args.accum_steps
        self.epochs = args.epochs

        self.eval_metric = args.eval_metric

        self.num_cpus = min(10, cpu_count() - 1) if cpu_count() > 1 else 1
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.pretrained_lm = {'pubmed': 'bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12',
                              'pubmed_mimic': 'bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12'}[args.bert_name]
        self.tokenizer = BertTokenizer.from_pretrained(self.pretrained_lm, do_lower_case=True)
        self.num_class = 2
        self.model = PICOSentClassModel.from_pretrained(self.pretrained_lm,
                                                        output_attentions=False,
                                                        output_hidden_states=False,
                                                        num_labels=self.num_class).to(self.device)
        self.train_data = self.create_dataset(self.corpus_path, self.train_file, loader_name="train.pt")
        self.dev_data = self.create_dataset(self.corpus_path, self.dev_file, loader_name="dev.pt")
        self.test_data = self.create_dataset(self.corpus_path, self.test_file, loader_name="test.pt")

        self.train_dataset = TensorDataset(self.train_data["input_ids"], self.train_data["attention_masks"],
                                           self.train_data["ground_labels"], self.train_data[self.label_query_name])
        self.dev_dataset = TensorDataset(self.dev_data["input_ids"], self.dev_data["attention_masks"],
                                         self.dev_data["ground_labels"], self.dev_data[self.label_query_name])
        self.test_dataset = TensorDataset(self.test_data["input_ids"], self.test_data["attention_masks"],
                                          self.test_data["ground_labels"], self.test_data[self.label_query_name])

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=True)
        self.dev_loader = DataLoader(self.dev_dataset, batch_size=self.eval_batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.eval_batch_size, shuffle=False)

    # convert a list of strings to token ids
    def encode(self, sents):
        encoded_dict = self.tokenizer.batch_encode_plus(sents,
                                                        is_split_into_words=True,
                                                        add_special_tokens=True,
                                                        max_length=self.max_len,
                                                        padding='max_length',
                                                        return_attention_mask=True,
                                                        truncation=True,
                                                        return_tensors='pt')
        input_ids = encoded_dict['input_ids']
        attention_masks = encoded_dict['attention_mask']
        return input_ids, attention_masks

    # convert dataset into tensors
    def create_dataset(self, dataset_dir, text_file, loader_name):
        loader_file = os.path.abspath(os.path.join(dataset_dir, loader_name))
        if os.path.exists(loader_file):
            print(f"Loading encoded texts from {loader_file}")
            data = torch.load(loader_file)
        else:
            print(f"Reading texts from {os.path.abspath(os.path.join(dataset_dir, text_file))}")
            with open(os.path.abspath(os.path.join(dataset_dir, text_file)), encoding="utf-8") as corpus:
                docs = [json.loads(doc) for doc in corpus]
            tokens = [doc['token'] for doc in docs]
            ground_labels = [doc['sent_ground'] for doc in docs]
            agg_labels = [doc['sent_agg'] for doc in docs]
            major_labels = [doc['sent_major'] for doc in docs]
            minor_labels = [doc['sent_minor'] for doc in docs]
            print(f"Converting texts into tensors.")
            chunk_size = ceil(len(docs) / self.num_cpus)
            chunks = [tokens[x:x+chunk_size] for x in range(0, len(docs), chunk_size)]
            results = Parallel(n_jobs=self.num_cpus)(delayed(self.encode)(sents=chunk) for chunk in chunks)
            input_ids = torch.cat([result[0] for result in results])
            attention_masks = torch.cat([result[1] for result in results])

            ground_labels = [int(label) for label in ground_labels]
            ground_labels = torch.tensor(ground_labels)
            agg_labels = [int(label) for label in agg_labels]
            agg_labels = torch.tensor(agg_labels)
            major_labels = [int(label) for label in major_labels]
            major_labels = torch.tensor(major_labels)
            minor_labels = [int(label) for label in minor_labels]
            minor_labels = torch.tensor(minor_labels)
            data = {"input_ids": input_ids, "attention_masks": attention_masks,
                    "ground_labels": ground_labels,
                    "agg_labels": agg_labels,
                    "major_labels": major_labels,
                    "minor_labels": minor_labels}

            print(f"Saving encoded texts into {loader_file}")
            torch.save(data, loader_file)
        return data

    def sentence_classification(self, epochs=5, loader_name="sentence_classification_model.pt"):
        model = self.model
        total_steps = len(self.train_loader) * epochs / self.accum_steps
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1 * total_steps,
                                                    num_training_steps=total_steps)
        top_metric = 0.0
        try:
            for i in range(epochs):
                model.train()
                total_train_loss = 0
                print(f"Epoch {i+1}:")
                torch.cuda.empty_cache()
                model.zero_grad()
                for j, batch in enumerate(tqdm(self.train_loader)):
                    input_ids = batch[0].to(self.device)
                    attention_mask = batch[1].to(self.device)
                    crowd_labels = batch[3].to(self.device)

                    loss, _, _, _ = model(input_ids,
                                          attention_mask=attention_mask,
                                          labels=crowd_labels)
                    loss /= self.accum_steps
                    total_train_loss += loss.item()
                    loss.backward()
                    if (j+1) % self.accum_steps == 0:
                        # Clip the norm of the gradients to 1.0.
                        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        scheduler.step()
                        model.zero_grad()
                avg_train_loss = torch.tensor([total_train_loss / len(self.train_loader) * self.accum_steps])
                print(f"Average training loss: {avg_train_loss.item()}")
                _, metric2 = self.inference(model, self.dev_loader, return_type=self.eval_metric)
                print(f"Develop crowd acc: {metric2}")
                if metric2 > top_metric:
                    top_metric = metric2
                    metric1, metric2 = self.inference(model, self.test_loader, return_type=self.eval_metric)
                    print(f"Test ground {self.eval_metric}: {metric1}, Test crowd {self.eval_metric}: {metric2}")
                    all_input_ids, all_input_mask, all_preds, all_last_hiddent_states, all_pooled_outputs \
                        = self.inference(model, self.test_loader, return_type="data")
                    loader_file = os.path.abspath(os.path.join(self.output_path, loader_name))
                    torch.save(model.state_dict(), loader_file)
                    torch.save(all_input_ids, os.path.abspath(os.path.join(self.output_path, 'all_input_ids.pt')))
                    torch.save(all_input_mask, os.path.abspath(os.path.join(self.output_path, 'all_input_mask.pt')))
                    torch.save(all_preds, os.path.abspath(os.path.join(self.output_path, 'all_preds.pt')))
                    torch.save(all_last_hiddent_states, os.path.abspath(os.path.join(self.output_path, 'all_last_hiddent_states.pt')))
                    torch.save(all_pooled_outputs, os.path.abspath(os.path.join(self.output_path, 'all_pooled_outputs.pt')))

        except RuntimeError as err:
            self.cuda_mem_error(err, "train")

    # use a model to do inference on a dataloader
    def inference(self, model, dataset_loader, return_type):
        if return_type == "data":
            all_input_ids = []
            all_input_mask = []
            all_preds = []
            all_last_hiddent_states = []
            all_pooled_outputs = []
        elif return_type == 'f1':
            pred_labels = []
            truth_labels = []
            crowd_labels = []
        elif return_type == "pred":
            pred_labels = []
        model.eval()
        try:
            for batch in dataset_loader:
                with torch.no_grad():
                    input_ids = batch[0].to(self.device)
                    attention_mask = batch[1].to(self.device)
                    loss, logits, last_hidden_states, pooled_output = model(input_ids,
                                                                            attention_mask=attention_mask)

                    if return_type == "data":
                        all_input_ids.append(input_ids)
                        all_input_mask.append(attention_mask)
                        all_preds.append(nn.Softmax(dim=-1)(logits))
                        all_last_hiddent_states.append(last_hidden_states)
                        all_pooled_outputs.append(pooled_output)
                    elif return_type == 'f1':
                        pred_labels.append(torch.argmax(logits, dim=-1).cpu())
                        truth_labels.append(batch[2])
                        crowd_labels.append(batch[3])
                    elif return_type == "pred":
                        pred_labels.append(torch.argmax(logits, dim=-1).cpu())
            if return_type == "data":
                all_input_ids = torch.cat(all_input_ids, dim=0).cpu()
                all_input_mask = torch.cat(all_input_mask, dim=0).cpu()
                all_preds = torch.cat(all_preds, dim=0).cpu()
                all_last_hiddent_states = torch.cat(all_last_hiddent_states, dim=0).cpu()
                all_pooled_outputs = torch.cat(all_pooled_outputs, dim=0).cpu()
                return all_input_ids, all_input_mask, all_preds, all_last_hiddent_states, all_pooled_outputs
            elif return_type == 'f1':
                pred_labels = torch.cat(pred_labels, dim=0).tolist()
                truth_labels = torch.cat(truth_labels, dim=0).tolist()
                crowd_labels = torch.cat(crowd_labels, dim=0).tolist()
                metric1 = f1_score(truth_labels, pred_labels)[-1][-1]
                metric2 = f1_score(crowd_labels, pred_labels)[-1][-1]
                return metric1, metric2
            elif return_type == "pred":
                pred_labels = torch.cat(pred_labels, dim=0)
                return pred_labels
        except RuntimeError as err:
            self.cuda_mem_error(err, "eval")

    # print error message based on CUDA memory error
    def cuda_mem_error(self, err, mode):
        print(err)
        if "CUDA out of memory" in str(err):
            if mode == "eval":
                print(f"Your GPUs can't hold the current batch size for evaluation, try to reduce `--eval_batch_size`, current: {self.eval_batch_size}")
            else:
                print(f"Your GPUs can't hold the current batch size for training, try to reduce `--train_batch_size`, current: {self.train_batch_size}")
        sys.exit(1)