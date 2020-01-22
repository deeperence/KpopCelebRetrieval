# 유틸리티 함수들을 정의하는 코드입니다.
import os, torch, random, re
from torch.backends import cudnn
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns

def MinMaxScaler (x):
    _max = x.max()
    _min = x.min()
    _denom = _max - _min
    return (x - _min) / _denom

def l2_normalize(v):
    norm = np.expand_dims(np.linalg.norm(v, axis=1), axis=1)
    if np.any(norm == 0):
        return v
    return v / norm

def fix_seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def to_np(t):
    return t.cpu().detach().numpy()

def soft_voting(probs):
    _arrs = [probs[key] for key in probs]
    return np.mean(np.mean(_arrs, axis=1), axis=0)

def pick_best_score(result1, result2):
    if result1['best_score'] < result2['best_score']:
        return result2
    else:
        return result1

def pick_best_loss(result1, result2):
    if result1['best_loss'] < result2['best_loss']:
        return result1
    else:
        return result2

def listDiff(li1, li2):
    return (list(set(li1) - set(li2)))

def remove_legacyModels(path):
    entire_file_list = os.listdir(path)
    best_score = 0
    best_model = None

    for filename in entire_file_list:
        if float(filename.split('-')[5]) > float(best_score):
            best_score = float(filename.split('-')[5])
            best_model = "".join(list(filename))
    print('Current best model:', str(best_model) + ' / score:', str(best_score))

    # best score model 제외한 나머지 모델 삭제
    del_model_list = listDiff(entire_file_list, [best_model])

    for model in del_model_list:
        os.remove(os.path.join(path, model))

def centerize(v1, v2):
    concat = np.concatenate([v1, v2], axis=0)
    center = np.mean(concat, axis=0)
    return v1-center, v2-center

