import os
from PIL import Image
import torchvision
import torchvision as vision
from augmentation.cutout import Cutout
from augmentation.augmentationPolicies import CIFAR10Policy
from backbones.efficientnet import EfficientNet

def custom_transforms_type1(target_size):
    data_transforms = {
        'train': torchvision.transforms.Compose([
            torchvision.transforms.Resize((target_size, target_size)),  # target_size 해상도로 이미지를 리사이즈합니다.
            torchvision.transforms.RandomHorizontalFlip(),  # 이미지를 랜덤하게 수평 반전시킵니다.
            torchvision.transforms.RandomRotation(20),  # 이미지를 랜덤하게 rotate합니다. 인자는 각도를 의미합니다.
            CIFAR10Policy(),
            torchvision.transforms.ToTensor(),  # torch tensor로 변환해 줍니다.
            torchvision.transforms.Normalize(  # 이미지를 z-score standardize해줍니다.
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
        ]),
        'validation': torchvision.transforms.Compose([
            torchvision.transforms.Resize((target_size, target_size)),
            torchvision.transforms.RandomResizedCrop(target_size, scale=(0.8, 1.0)),
            # 주어진 이미지를 인자로 넣어 준 범위 내에서 임의의 크기 및 종횡비로 자르는 함수입니다.
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
        ]),
        'test': torchvision.transforms.Compose([
            torchvision.transforms.Resize((target_size, target_size)),
            torchvision.transforms.RandomResizedCrop(target_size, scale=(0.8, 1.0)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
        ]),
        'tta': vision.transforms.Compose([
            vision.transforms.Resize(target_size),
            vision.transforms.RandomHorizontalFlip(),
            vision.transforms.RandomRotation(20),
            CIFAR10Policy(),
            vision.transforms.ToTensor(),
            vision.transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
        ]),
    }
    return data_transforms

def custom_transforms_type2(model_name, target_size):
    if 'efficient' in model_name:
        image_size = EfficientNet.get_image_size(model_name)
        print(image_size)
        data_transforms = {
            'train': vision.transforms.Compose([
                vision.transforms.Resize((target_size, target_size)),
                vision.transforms.RandomHorizontalFlip(),
                vision.transforms.RandomRotation(20),
                CIFAR10Policy(),
                vision.transforms.ToTensor(),
                Cutout(n_holes=1, length=image_size // 4),
                vision.transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225])
            ]),
            'validation': vision.transforms.Compose([
                vision.transforms.Resize((target_size, target_size)),
                vision.transforms.RandomResizedCrop(target_size, scale=(0.8, 1.0)),
                vision.transforms.RandomHorizontalFlip(),
                vision.transforms.ToTensor(),
                vision.transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225])
            ]),
            'test': vision.transforms.Compose([
                vision.transforms.Resize((target_size, target_size)),
                vision.transforms.RandomResizedCrop(target_size, scale=(0.8, 1.0)),
                vision.transforms.ToTensor(),
                vision.transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225])
            ]),
            'tta': vision.transforms.Compose([
                vision.transforms.Resize(target_size),
                vision.transforms.RandomHorizontalFlip(),
                vision.transforms.RandomRotation(20),
                CIFAR10Policy(),
                vision.transforms.ToTensor(),
                vision.transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225])
            ]),
        }
        return data_transforms
    else:
        data_transforms = custom_transforms_type1(target_size)['test']
        return data_transforms

class TrainDataset():
    def __init__(self, TRAIN_IMAGE_PATH, df, transforms=None):
        self.df = df
        self.train_data_path = TRAIN_IMAGE_PATH
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.train_data_path, self.df['img_file'][idx])).convert("RGB")
        label = self.df['new_class'][idx]
        if self.transforms:
            image = self.transforms(image)

        return image, int(label)

class TestDataset():
    def __init__(self, TEST_IMAGE_PATH, df, transforms=None):
        self.df = df
        self.test_data_path = TEST_IMAGE_PATH
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.test_data_path, self.df['img_file'][idx])).convert("RGB")
        if self.transforms:
            image = self.transforms(image)

        return image