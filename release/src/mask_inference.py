import json
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer
import os
from tqdm import tqdm
from model import PICOSentClassModel
import argparse
import numpy as np
from .inference import get_final_inference_result, decode_topk


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def data_load(dataset_dir, bert_name):
    '''
    data = {"input_ids": input_ids, "attention_masks": attention_masks,
            "ground_labels": ground_labels,
            "agg_labels": agg_labels,
            "major_labels": major_labels,
            "minor_labels": minor_labels}
    '''

    with open(os.path.abspath(os.path.join(dataset_dir, 'test.json')), encoding="utf-8") as corpus:
        docs = [json.loads(doc) for doc in corpus]

    data = torch.load(os.path.abspath(os.path.join(dataset_dir, 'test.pt')))
    print(data['input_ids'], data['input_ids'].shape)
    pretrained_lm = {'pubmed': 'bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12',
                     'pubmed_mimic': 'bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12'}[bert_name]
    tokenizer = BertTokenizer.from_pretrained(pretrained_lm, do_lower_case=True)
    decoded_tokens = [tokenizer.convert_ids_to_tokens(data['input_ids'][i], skip_special_tokens=True) for i in
                      range(data['input_ids'].shape[0])]
    return docs, data, decoded_tokens


def tokenid2bertid(tokens, tokenizer, add_special_tokens=False):
    ans = [tokenizer.encode(t, add_special_tokens=False) for t in tokens]
    ans = [t for t in ans if len(t)>0]
    token_pid2bert_pid_dict = {}
    bert_pid2token_pid = []
    start_pos = 0 if not add_special_tokens else 1
    for i, ids in enumerate(ans):
        if len(ids) == 0:
            continue
        token_pid2bert_pid_dict[i] = []
        for _ in ids:
            token_pid2bert_pid_dict[i].append(start_pos)
            start_pos += 1
        bert_pid2token_pid.extend([i]*len(ids))
    token_pid2bert_pid_dict[len(token_pid2bert_pid_dict)] = [start_pos]
    if add_special_tokens:
        bert_pid2token_pid = [-1] + bert_pid2token_pid + [-1]
    return token_pid2bert_pid_dict, bert_pid2token_pid, ans


def token_mask_generator(tokens, tokenizer, max_mask_size=10, add_special_tokens=False, max_len=512, batch_size=64):
    token_pid2bert_pid_dict, bert_pid2token_pid, nested_tokenization_res = tokenid2bertid(tokens, tokenizer, add_special_tokens)

    num_token = len(token_pid2bert_pid_dict) - 1
    max_mask_size = min(max_mask_size, num_token)
    print(len(nested_tokenization_res), num_token, max_mask_size, token_pid2bert_pid_dict)
    mask_id = tokenizer.mask_token_id
    mask_pos = []
    mask_range = []
    for mask_len in range(1, max_mask_size+1):
        for token_id in range(num_token-mask_len+1):
            tmp = np.zeros(max_len)
            print(token_id, token_id+mask_len)
            mask_start = token_pid2bert_pid_dict[token_id][0]
            mask_end = token_pid2bert_pid_dict[token_id+mask_len][0]
            tmp[mask_start:mask_end] = 1
            mask_pos.append(tmp)
            mask_range.append((token_id, token_id+mask_len))
    mask_chunks = [torch.from_numpy(np.stack(mask_pos[x:x + batch_size])) > 0 for x in range(0, len(mask_pos), batch_size)]
    # mask_pos = torch.from_numpy(np.stack(mask_pos)) > 0
    return token_pid2bert_pid_dict, bert_pid2token_pid, nested_tokenization_res, mask_chunks, mask_range, mask_id


def mask_prediction_inference(model, data_loader, docs, tokenizer, max_mask_size=10, add_special_tokens=False, max_len=512, batch_size=64):
    model.eval()
    result = []
    for i, batch in enumerate(tqdm(data_loader)):
        with torch.no_grad():
            torch.cuda.empty_cache()
            input_ids = batch[0]
            attention_mask = batch[1]
            # ground_labels = batch[2].to(device)
            # crowd_labels = batch[3].to(device)
            input_lens = attention_mask.sum(-1)
            input_ids = input_ids[:, :input_lens]
            attention_mask = attention_mask[:, :input_lens]

            orgin_logits = model(input_ids.to(device), attention_mask=attention_mask.to(device))[1].detach().cpu().numpy()

            # get input masks
            doc = docs[i]
            tokens = doc['token']
            token_pid2bert_pid_dict, bert_pid2token_pid, nested_tokenization_res, mask_chunks, mask_range, mask_id = token_mask_generator(tokens, tokenizer, max_mask_size, add_special_tokens, max_len, batch_size)
            # gather logit for chunks
            logits = []
            for mask_chunk in mask_chunks:
                mask_chunk = mask_chunk[:, :input_lens]
                num_masks = mask_chunk.shape[0]
                tmp_input_ids = input_ids.repeat(num_masks, 1)
                tmp_attention_mask = attention_mask.repeat(num_masks, 1)

                # mask out category indicative words
                tmp_input_ids[mask_chunk] = mask_id
                tmp_logits = model(tmp_input_ids.to(device), attention_mask=tmp_attention_mask.to(device))[1].detach().cpu().numpy()     # mask_num, 2
                logits.append(tmp_logits)
            logits = np.concatenate(logits, axis=0)
            min_positive_index = np.argmin(logits[:, 1])
            min_positive_score = logits[min_positive_index, 1]
            # min_positive_score, min_positive_index = torch.min(logits[:, 1], dim=0)
            result.append({'inference_mask': mask_range[min_positive_index],
                           'min_positive_index': int(min_positive_index),
                           'prediction': int(orgin_logits[0][1] > orgin_logits[0][0]),
                           'orgin_score': orgin_logits.tolist(),
                           'min_positive_score': float(min_positive_score),
                           'mask_range': mask_range,
                           'logits': logits.tolist()})
    return result


def main():
    parser = argparse.ArgumentParser(description='Infer most possible span with trained text classification model.')
    parser.add_argument('--corpus_path', default='../data/ebm_pico_p/', help='the path to the corpus folder.')
    parser.add_argument('--output_path', default='../exps/pubmed/ebm_pico_p/', help='the path to the ouput folder.')
    parser.add_argument('--label_name', default='aggregation', help='the sentence label generation type',
                        choices=['major', 'minor', 'aggregation'])
    parser.add_argument('--bert_name', default='pubmed', help='the pre-trained bert model name',
                        choices=['pubmed', 'pubmed_mimic'])
    parser.add_argument('--eval_batch_size', type=int, default=64,
                        help='batch size per GPU for evaluation; bigger batch size makes training faster')
    parser.add_argument('--max_len', type=int, default=512,
                        help='length that documents are padded/truncated to')
    parser.add_argument('--max_mask_len', type=int, default=10,
                        help='the max num of mask token length')

    args = parser.parse_args()
    print(args)
    # load tokenizer
    pretrained_lm = {'pubmed': 'bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12',
                     'pubmed_mimic': 'bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12'}[args.bert_name]
    tokenizer = BertTokenizer.from_pretrained(pretrained_lm, do_lower_case=True)
    # load data
    docs, pt_data, decoded_tokens = data_load(args.corpus_path, args.bert_name)
    label_query_name = {'aggregation': 'agg_labels',
                        'major': 'major_labels',
                        'minor': 'minor_labels'}[args.label_name]
    data_dataset = TensorDataset(pt_data["input_ids"], pt_data["attention_masks"],
                                 pt_data["ground_labels"], pt_data[label_query_name])
    data_loader = DataLoader(data_dataset, batch_size=1, shuffle=False)
    # load model
    model_folder = os.path.join(args.output_path, args.label_name, '_pool_', 'f1')
    model = PICOSentClassModel.from_pretrained(pretrained_lm,
                                               output_attentions=False,
                                               output_hidden_states=False,
                                               num_labels=2).to(device)
    model.load_state_dict(torch.load(os.path.abspath(os.path.join(model_folder, "sentence_classification_model.pt"))))
    inferred_result = mask_prediction_inference(model, data_loader, docs, tokenizer, max_mask_size=args.max_mask_len,
                                                add_special_tokens=True, max_len=args.max_len, batch_size=args.eval_batch_size)
    with open(os.path.abspath(os.path.join(args.output_path, args.label_name, '_pool_', 'f1', 'direct_infer_result.json')), 'w') as fout:
        json.dump(inferred_result, fout)
    final_infer_result = get_final_inference_result(docs, inferred_result, decode_topk)
    with open(os.path.abspath(os.path.join(args.output_path, args.label_name, '_pool_', 'f1', 'final_infer_result.json')), 'w') as fout:
        json.dump(final_infer_result, fout)

if __name__ == '__main__':
    main()
