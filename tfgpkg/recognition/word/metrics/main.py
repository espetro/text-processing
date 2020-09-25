from typing import List

import logging

logging.basicConfig(
    format='%(levelname)s(%(filename)s:%(lineno)d): %(message)s')

def wer(true_sentence: str, pred_sentence: str):
    """"""
    pass

def cer(true_word: str, pred_word: str):
    """"""
    pass

def levenshtein(u, v):
    prev = None
    curr = [0] + list(range(1, len(v) + 1))
    # Operations: (SUB, DEL, INS)
    prev_ops = None
    curr_ops = [(0, 0, i) for i in range(len(v) + 1)]
    for x in range(1, len(u) + 1):
        prev, curr = curr, [x] + ([None] * len(v))
        prev_ops, curr_ops = curr_ops, [(0, x, 0)] + ([None] * len(v))
        for y in range(1, len(v) + 1):
            delcost = prev[y] + 1
            addcost = curr[y - 1] + 1
            subcost = prev[y - 1] + int(u[x - 1] != v[y - 1])
            curr[y] = min(subcost, delcost, addcost)
            if curr[y] == subcost:
                (n_s, n_d, n_i) = prev_ops[y - 1]
                curr_ops[y] = (n_s + int(u[x - 1] != v[y - 1]), n_d, n_i)
            elif curr[y] == delcost:
                (n_s, n_d, n_i) = prev_ops[y]
                curr_ops[y] = (n_s, n_d + 1, n_i)
            else:
                (n_s, n_d, n_i) = curr_ops[y - 1]
                curr_ops[y] = (n_s, n_d, n_i + 1)
    return curr[len(v)], curr_ops[len(v)]

def compute_stats(reference: List[str], hypothesis: List[str]):
    """"""
    wer_s, wer_i, wer_d, wer_n = 0, 0, 0, 0
    cer_s, cer_i, cer_d, cer_n = 0, 0, 0, 0
    sen_err = 0
    for n in range(len(reference)):
        # update CER statistics
        _, (s, i, d) = levenshtein(reference[n], hypothesis[n])
        cer_s += s
        cer_i += i
        cer_d += d
        cer_n += len(reference[n])
        # update WER statistics
        _, (s, i, d) = levenshtein(reference[n].split(), hypothesis[n].split())
        wer_s += s
        wer_i += i
        wer_d += d
        wer_n += len(reference[n].split())
        # update SER statistics
        if s + i + d > 0:
            sen_err += 1

    if cer_n > 0:
        result_cer = (100.0 * (cer_s + cer_i + cer_d)) / cer_n
        result_wer = (100.0 * (wer_s + wer_i + wer_d)) / wer_n
        result_ser = (100.0 * sen_err) / len(reference)
        
        return result_cer, result_wer, result_ser
