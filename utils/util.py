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
    norm = np.linalg.norm(v)
    if norm == 0:
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

def remove_legacyModels(path):
    entire_file_list = os.listdir(path)
    for filename in entire_file_list:
        # best score 갱신
        if model in filename and float(filename.split('-')[-2]) > float(fold_best_model_score):
            fold_best_model_score = filename.split('-')[-2]
            fold_best_model_name = filename
    best_models.append(fold_best_model_name)
    print(fold + ' best model:', fold_best_model_name + ' / cv score:', fold_best_model_score)

    # best score model 제외한 나머지 모델 삭제
    del_model_list = list(set(entire_file_list) - set(best_models))
    print('Current best models: ', [filename for filename in best_models if filename.endswith(filename.split('.')[-1])])
    for file in del_model_list:
        os.remove(os.path.join(path, file))