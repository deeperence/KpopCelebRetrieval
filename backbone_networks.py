from torchvision import models
import torch.nn as nn
from backbones.efficientnet import EfficientNet

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            # False = 역전파 중 해당 Tensor에 대한 Gradient를 계산하지 않을 것임을 의미
            param.requires_grad = False

def initialize_model(args, model_name, feature_extract=True, use_pretrained=True, num_classes=None):
    '''
    Define network. Initialize these variables which will be set in this if statement.
    Each of these variables is model specific. (See also https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html)
    - If 'feature_extract=True', only update the newly stacked layer params.
    - If 'feature_extract=False', whole model params will be updated. (Including newly stacked layer params)
    # Torchvision 모델의 경우 모델 선언 부분에 num_classes를 지정해주지 않아도 됨. (.in_features 함수로 자동 지정)
    '''
    model_ft = None

    # ImageNet-1000 pretrained Efficientnet (From third-party repository: https://github.com/lukemelas/EfficientNet-PyTorch)
    if model_name == 'efficientnet-b0':
        return EfficientNet.from_pretrained('efficientnet-b0', num_classes = num_classes)
    elif model_name == 'efficientnet-b1':
        return EfficientNet.from_pretrained('efficientnet-b1', num_classes = num_classes)
    elif model_name == 'efficientnet-b2':
        return EfficientNet.from_pretrained('efficientnet-b2', num_classes = num_classes)
    elif model_name == 'efficientnet-b3':
        return EfficientNet.from_pretrained('efficientnet-b3', num_classes = num_classes)
        # return EfficientNet.from_scratch(model_name='efficientnet-b3', num_classes=num_classes)
    elif model_name == 'efficientnet-b4':
        return EfficientNet.from_pretrained('efficientnet-b4', num_classes = num_classes)
    elif model_name == 'efficientnet-b5':
        EfficientNet.from_pretrained('efficientnet-b5', num_classes = num_classes)
    else:
        print("Wrong define model parameter input.")
        raise ValueError