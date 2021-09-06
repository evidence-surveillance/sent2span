import numpy as np

def span_checker(span, remove_spans):
    span_start, span_end = span
    flag = False
    for i in range(span_start + 1, span_end):
        if tuple((span_start, i)) in remove_spans and tuple((i, span_end)) in remove_spans:
            flag = True
            break
    return flag


def nested_span_remover(res):
    original_score = res['orgin_score'][0]
    true_prediction = original_score[1]
    remove_spans = set()
    remove_span_idx = set()
    for i, (span, logit) in enumerate(zip(res['mask_range'], res['logits'])):
        if span[1] - span[0] == 1:
            if logit[1] > true_prediction:
                remove_span_idx.add(i)
                remove_spans.add(tuple(span))
        else:
            if span_checker(span, remove_spans):
                remove_span_idx.add(i)
                remove_spans.add(tuple(span))
    return remove_spans, remove_span_idx


def decode_topk(span_result, k=2):
    assert k > 0
    masks = span_result['mask_range']
    logits = span_result['logits']
    origin_score = np.array(span_result['orgin_score'][0])

    span_result_list = sorted(
        [(m, logit, origin_score[0]-logit[0], origin_score[1]-logit[1])
         for m, logit in zip(masks, logits)], key=lambda x: x[-1], reverse=True)
    if len(span_result_list) <= k:
        result = span_result_list
    else:
        tmp_fill = np.full(max(m[1] for m in masks), 0)
        result = []
        while span_result_list and len(result) < k:
            tmp_span = span_result_list.pop(0)
            if sum(tmp_fill[tmp_span[0][0]:tmp_span[0][1]]) == 0:
                result.append(tmp_span)
                tmp_fill[result[-1][0][0]:result[-1][0][1]] = 1

    return [r for r in result if r[2] < r[3]]


def get_final_inference_result(dataset, infer_res, decode_alg):
    final_infer_result = []
    for i, (sent, res) in enumerate(zip(dataset, infer_res)):
        remove_span, remove_span_idx = nested_span_remover(res)

        alg_infer_res = []
        if res['prediction'] and len(remove_span_idx) < len(res['mask_range']):
            tmp_res = {'mask_range': [m for rix, m in enumerate(res['mask_range']) if rix not in remove_span_idx],
                       'logits': [m for rix, m in enumerate(res['logits']) if rix not in remove_span_idx],
                       'orgin_score': res['orgin_score']
                       }
            alg_infer_res = decode_alg(tmp_res)
        final_infer_result.append(alg_infer_res)

    return final_infer_result