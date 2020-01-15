import numpy as np
from tqdm import tqdm
import torch, time, os
from torch.utils import data
from torchvision import datasets, transforms

current_path = os.getcwd()
IMG_PATH = os.path.join(current_path, '../../dataset')

def data_loader (root_path):
    t = time.time()
    print('Data loading...')
    data = [] # data path 저장을 위한 변수
    labels=[] # 테스트 id 순서 기록
    ## 하위 데이터 path 읽기
    for dir_name,_,_ in os.walk(root_path):
        try:
            data_id = dir_name.split('/')[-1]
            int(data_id)
        except: pass
        else:
            data.append(np.load(dir_name+'/mammo.npz')['arr_0'])
            labels.append(int(data_id[0]))
    data = np.array(data) ## list to numpy
    labels = np.array(labels) ## list to numpy
    print('Dataset Reading Success \n Reading time',time.time()-t,'sec')
    print('Dataset:',data.shape,'np.array.shape(files, views, width, height)')
    print('Labels:', labels.shape, 'each of which 0~2')
    return data, labels

def calcMeanStd(dataloader):
    """Compute the mean and sd in an online fashion
        Var[x] = E[X^2] - E^2[X]
    """
    tbar = tqdm(dataloader)
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for images, _ in tbar:
        b, c, h, w = images.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images ** 2, dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)

        cnt += nb_pixels

    return fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)

def main():
    # Build pytorch dataset and batch dataloader
    dataset = datasets.ImageFolder(root=IMG_PATH,
                                   transform=transforms.Compose([
                                   transforms.Resize((224, 224)),
                                   transforms.ToTensor()]))
    batch_size = len(dataset) # OOM 에러 발생 시 배치사이즈를 하드코딩해서 넣기.
    dataloader = data.DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=False)
    print("\nlength of dataloader : ", len(dataloader) * batch_size)
    entire_mean, entire_std = calcMeanStd(dataloader)
    print("Entire dataset`s mean and std(mean/std) per channels : ", entire_mean, entire_std)

if __name__ == "__main__":
    main()