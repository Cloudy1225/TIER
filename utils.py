import datetime
import random
import time

import numpy as np
import pytz
import torch
from sklearn.metrics import accuracy_score, f1_score


MODEL_PATHs = {
    # LM
    "MiniLM": "sentence-transformers/all-MiniLM-L6-v2",
    "SentenceBert": "sentence-transformers/multi-qa-distilbert-cos-v1",
    "e5-large": "intfloat/e5-large-v2",
    "roberta": "sentence-transformers/all-roberta-large-v1",

    # LLM
    "Qwen-3B": "Qwen/Qwen2.5-3B-Instruct",
    "Qwen-7B": "Qwen/Qwen2.5-7B-Instruct",
    "Qwen-14B": "Qwen/Qwen2.5-14B-Instruct",
    "Qwen-32B": "Qwen/Qwen2.5-32B-Instruct",
    "Mistral-7B": "mistralai/Mistral-7B-Instruct-v0.2",
    "Llama-8B": "meta-llama/Llama-3.1-8B-Instruct",
}


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)  # gpu
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def array_mean_std(numbers):
    array = np.array(numbers)
    return np.round(np.mean(array), 3), np.round(np.std(array), 3)


def compute_acc_and_f1(pred, ground_truth):
    accuracy = accuracy_score(ground_truth, pred) * 100.0
    macro_f1 = f1_score(ground_truth, pred, average="macro") * 100.0
    weighted_f1 = f1_score(ground_truth, pred, average="weighted") * 100.0

    return round(accuracy, 2), round(macro_f1, 2), round(weighted_f1, 2)

def get_cur_time(timezone='Asia/Shanghai', t_format='%m-%d %H:%M:%S'):
    return datetime.datetime.fromtimestamp(int(time.time()), pytz.timezone(timezone)).strftime(t_format)
