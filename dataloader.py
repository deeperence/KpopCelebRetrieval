import os
from PIL import Image
import torchvision as vision
from augmentation.augmentationPolicies import CIFAR10Policy
from backbones.efficientnet import EfficientNet

def custom_transforms(model_name, target_size):
    image_size = EfficientNet.get_image_size(model_name)
    print(model_name + ' image size:',image_size)
    data_transforms = {
        'train': vision.transforms.Compose([
            vision.transforms.Resize((target_size, target_size)),
            vision.transforms.RandomHorizontalFlip(),
            vision.transforms.RandomRotation(10),
            CIFAR10Policy(),
            vision.transforms.ToTensor(),
            vision.transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
        ]),
        'validation': vision.transforms.Compose([
            vision.transforms.Resize((target_size, target_size)),
            vision.transforms.RandomHorizontalFlip(),
            vision.transforms.ToTensor(),
            vision.transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
        ]),
        # 'test'는 reference와 query 이미지에 대해 수행됩니다.
        'test': vision.transforms.Compose([
            vision.transforms.Resize((target_size, target_size)),
            vision.transforms.ToTensor(),
            vision.transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
        ]),
    }
    return data_transforms

class TrainDataset():
    def __init__(self, TRAIN_IMAGE_PATH, df, transforms=None):
        self.df = df
        self.train_data_path = TRAIN_IMAGE_PATH
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.train_data_path, self.df['filename'][idx])).convert("RGB")
        label = self.df['class'][idx]
        if self.transforms:
            image = self.transforms(image)

        return image, int(label)

class TestDataset():
    def __init__(self, TEST_IMAGE_PATH, df, transforms=None):
        self.df = df
        self.test_data_path = TEST_IMAGE_PATH
        self.transforms = transforms

    def __len__(self):
        len(self.test_data_path)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.test_data_path, self.df['filename'][idx])).convert("RGB")
        image_path = os.path.join(self.test_data_path, self.df['filename'][idx])
        if self.transforms:
            image = self.transforms(image)

        return image_path, image