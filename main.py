import argparse, os, torch, warnings, time, gc
import numpy as np
from datetime import datetime
from torch import cuda
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.metrics import Evaluator
from lr_scheduler.CosineAnnealingWithRestartsLR import CosineAnnealingWithRestartsLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from lr_scheduler.lr_scheduler import defineLRScheduler
from optimizers.AdamW import AdamW
from optimizers.RAdam import RAdam
from tensorboardX import SummaryWriter
from sklearn.model_selection import StratifiedKFold
from torch.backends import cudnn
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
from torch import optim
from tqdm import tqdm
from utils import util
from sklearn.metrics import f1_score
from dataloader import TestDataset, TrainDataset, custom_transforms_type
from backbone_networks import initialize_model

class Trainer(object):
    def __init__(self, args, writer):
        self.args = args
        self.writer = writer # object for saving current status
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.df_train = pd.read_csv(os.path.join(args.dataset_path, 'chexcar_train.csv'))
        self.df_train['new_class'] = self.df_train['new_class'].astype('int')
        self.df_train.replace(196, 0, inplace=True)

        if args.mode == 'train':
            print('Training Start...')
            print('Total epoches:', args.epochs)
            stratified_folds = StratifiedShuffleSplit(n_splits=1, test_size=0.1)

            for (train_index, validation_index) in stratified_folds.split(self.df_train['img_file'], self.df_train['new_class']):
                df_train = self.df_train.iloc[train_index, :].reset_index()
                df_validation = self.df_train.iloc[validation_index, :].reset_index()

                # Load custom transforms
                self.transform = custom_transforms_type(model_name=args.backbone, target_size=args.image_size)

                # Load dataset for train
                train_dataset = TrainDataset(os.path.join(args.dataset_path, 'train_cropped'), df_train, transforms=self.transform['train'])
                train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

                # Load dataset for validation
                validation_dataset = TrainDataset(os.path.join(args.dataset_path, 'train_cropped'), df_validation, transforms=self.transform['validation'])
                validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False)

                print('Dataset class : ', self.args.class_num)
                print("Holdout Train dataset length: {} / Holdout Validation dataset length: {}".format(str(len(train_dataset)), str(len(validation_dataset))))

                # Define model
                model, evaluator, lr_scheduler, optimizer = self.define_network(args=args, length_train_dataloader=len(train_dataset))
                model.to(self.device)

                # Define Criterion
                criterion = nn.CrossEntropyLoss()  # buildLosses(cuda=args.cuda).build_loss(mode=args.loss_type)
                print('optimizer : ', type(optimizer), ' lr scheduler : ', type(lr_scheduler))

                # Train the model
                self.train_model(args=args, model = model, optimizer=optimizer, scheduler=lr_scheduler, criterion=criterion, train_loader=train_loader,
                                 validation_dataset=validation_dataset, validation_loader=validation_loader, weight_file_name='weight_best.pt')
                del(model, evaluator, lr_scheduler, optimizer)
                gc.collect()

        elif args.mode == 'test': # Test 모드로 설정 후 모델 infer 수행 및 score 계산
            '''
            모델은 Query image를 인풋으로 받고 각 reference images와 유사도를 비교한 후 query image와 유사한 순서대로 reference image filename을 정렬해 보여줍니다. 
            - Query image는 검색기에 테스트용으로 넣을 이미지입니다. 
            - Reference image는 query 이미지와 유사도 비교의 대상이 되는 이미지입니다.
            '''

            print('Inferring Start...')
            query_path = os.path.join(args.dataset_path, '/Image/Queries')
            reference_path = os.path.join(args.dataset_path, '/Image/References')
            model_path = args.model_savepath
            weight_file = [file for file in os.listdir(model_path) if file.endswith(".pt")] # 'model_savepath' 디렉토리 안의 .pt파일 조회
            db = [os.path.join(reference_path, path) for path in os.listdir(reference_path)] # 'reference_path' 디렉토리 안의 reference image 파일 이름을 db 리스트에 append합니다.

            queries = [v.split('/')[-1].split('.')[0] for v in os.listdir(query_path)] # 'query_path'의 각 파일들로부터 파일 이름만 남깁니다. (e.g. 'yooa1', 'yeji3', ...)
            db = [v.split('/')[-1].split('.')[0] for v in db] # 'reference_path'의 각 파일들로부터 파일 이름만 남깁니다.
            queries.sort()
            db.sort()

            transform = custom_transforms_type(model_name = args.backbone, target_size=args.image_sizes)
            query_dataset = TestDataset(query_path, df_test, transforms=transform['test'])
            query_loader = DataLoader(query_dataset, batch_size=1, shuffle=False)
            ref_dataset = TestDataset(reference_path, df_test, transforms=transform['test'])
            ref_loader = DataLoader(ref_dataset, batch_size=args., shuffle=False)

            model = initialize_model(args=None, model_name=args.backbone, num_classes=args.class_num)
            model.load_state_dict(torch.load(weight_file), strict=False)
            print('model loaded:', weight_file)
            model.eval()
            model.to('cuda')

            # Query image에 대한 feature vector를 학습이 다 된 모델로부터 추출합니다.
            for query_i, query_images in enumerate(query_loader):
                query_images = query_images.cuda()
                query_vecs = np.asarray(model(query_images).detach())

            # Reference image에 대한 feature vector를 학습이 다 된 모델로부터 추출합니다.
            for ref_i, ref_images in enumerate(ref_loader):
                ref_images = ref_images.cuda()
                reference_vecs =  np.asarray(model(ref_images).detach())



            # l2 normalization
            query_vecs = util.l2_normalize(query_vecs)
            reference_vecs = util.l2_normalize(reference_vecs)

            # Calculate cosine similarity
            sim_matrix = np.dot(query_vecs, reference_vecs.T)
            indices = np.argsort(sim_matrix, axis=1)
            indices = np.flip(indices, axis=1)

            retrieval_results = {}

            for (i, query) in enumerate(queries):
                ranked_list = [references[k] for k in indices[i]]
                ranked_list = ranked_list[:1000]

                retrieval_results[query] = ranked_list
            print('Retrieval done.')

            final_retrieval_list = list(zip(range(len(retrieval_results)), retrieval_results.items()))
            print(final_retrieval_list)

        else:
            print("wrong mode input.")
            raise NotImplementedError

    def define_network(self, args, length_train_dataloader):
        '''
        Define network.
        - If feature_extract=True, only update the newly stacked layer params.
        - If feature_extract=False, whole model params will be updated. (Including newly stacked layer params)
        - 'use_pretrained'(bool) decides whether using ImageNet-1000 pretrained weights.
        #'''

        # Note that below model doesn't have softmax layer.
        model = initialize_model(args = args, model_name=args.backbone, feature_extract=args.feature_extract, use_pretrained=True, num_classes=args.class_num)

        # Print parameters to be optimized/updated.
        print("Params to learn:")
        params_to_update = []
        try:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    params_to_update.append(param)
                    print("\t", name)
        except:
            pass

        # Define Evaluator (F1, Acc_class 등의 metric 계산을 정의한 클래스)
        evaluator = Evaluator(self.args.class_num)

        # choose scheduler
        if args.use_pretrained == True: # Whether using pretrained model for fine-tunning
            lr = 0.00001
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)  # weight_decay=0.000025
            if args.is_scheduler == True:
                if args.lr_scheduler == 'CosineAnnealingWithRestartsLR':
                    eta_min = 1e-6
                    T_max = 10
                    T_mult = 1
                    restart_decay = 0.97
                    lr_scheduler = CosineAnnealingWithRestartsLR(optimizer, T_max=T_max, eta_min=eta_min, T_mult=T_mult, restart_decay=restart_decay)
                else:
                    lr_scheduler = defineLRScheduler(args, optimizer, length_train_dataloader)
                    # lr_scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.1)
            elif args.is_scheduler == False:
                lr_scheduler = None
            else:
                print("Invalid args.lr_scheduler input.")
                raise NotImplementedError
        elif args.use_pretrained == False:
            # Define Optimizer
            if args.optim.lower() == 'sgd':
                optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)  # weight_decay=0.000025
            elif args.optim.lower() == 'adam':
                optimizer = optim.Adam(model.parameters(), lr=args.lr)
            elif args.optim.lower() == 'adamw':
                optimizer = AdamW(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
            elif args.optim.lower() == 'radam':
                optimizer = RAdam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
            else:
                print("Wrong optimizer args input.")
                raise NotImplementedError

            if args.is_scheduler == True:
                if args.lr_scheduler == 'CosineAnnealingWithRestartsLR':
                    eta_min = 1e-6
                    T_max = 10
                    T_mult = 1
                    restart_decay = 0.97
                    lr_scheduler = CosineAnnealingWithRestartsLR(optimizer, T_max=T_max, eta_min=eta_min, T_mult=T_mult, restart_decay=restart_decay)
                else:
                    lr_scheduler = defineLRScheduler(args, optimizer, length_train_dataloader)
                    # lr_scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.1)
            elif args.is_scheduler == False:
                lr_scheduler = None
            else:
                print("Invalid args.lr_scheduler input.")
                raise NotImplementedError
        else:
            print('Invalid args.use_pretrained input.')
            raise NotImplementedError

        return model, evaluator, lr_scheduler, optimizer

    def train_one_epoch(self, model, criterion, train_loader, optimizer):
        model.train()
        optimizer.zero_grad()
        train_loss = 0.
        for step, (inputs, targets) in tqdm(iterable=enumerate(train_loader), desc='Training Step: ', total=len(train_loader), leave=False):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item() / len(train_loader)
            epoch_lr = util.get_lr(optimizer)
            self.writer.add_scalar('train/batch_loss', loss, step)
            self.writer.add_scalar('batch/learning_rate', epoch_lr, step)

        return train_loss

    def validation(self, model, criterion, validation_dataset, valid_loader, df_valid):
        model.eval()
        valid_preds = np.zeros((len(validation_dataset), self.args.class_num))
        val_loss = 0.

        with torch.no_grad():
            for i, (inputs, targets) in tqdm(iterable=enumerate(valid_loader), desc='Validation Step: ', total=len(valid_loader), leave=False):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs).detach()
                loss = criterion(outputs, targets)
                valid_preds[i * self.args.batch_size: (i+1) * self.args.batch_size] = outputs.cpu().numpy()
                val_loss += loss.item() / len(valid_loader)
            y_pred = np.argmax(valid_preds, axis=1)
            val_score = f1_score(df_valid['new_class'].values, y_pred, average='micro')

        return val_loss, val_score

    def train_model(self, args, model, optimizer, scheduler, criterion, train_loader, validation_dataset, validation_loader, weight_file_name='weight_best.pt'):
        model.train() # Train모드로 전환
        self.args = args

        # Set variables for store train results
        train_result = {}
        train_result['weight_file_name'] = weight_file_name
        best_score = 0.
        lrs = []
        score = []

        for epoch in tqdm(range(args.epochs)):
            start_time = time.time()
            now = datetime.now()
            train_loss = self.train_one_epoch(model, criterion, train_loader, optimizer)
            val_loss, val_score = self.validation(args, model, criterion, validation_dataset, validation_loader)
            score.append(val_score)
            self.writer.add_scalar('train/epoch_loss', train_loss, epoch)
            self.writer.add_scalar('val/epoch_loss', val_loss, epoch)
            self.writer.add_scalar('val/epoch_F1score', val_score, epoch)

            # model save (score or loss?)
            if args.cv_checkpoint:
                if val_score > best_score:
                    best_score = val_score
                    train_result['best_epoch'] = epoch + 1
                    train_result['best_score'] = round(best_score, 5)
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'lr': util.get_lr(optimizer),},
                        args.checkname + 'epoch' + str(epoch) + '-' + str(round(val_score,4))+'-'+ weight_file_name)
                    print('\n')
                    print('Saved model name: ', args.checkname + '-epoch' + str(epoch) + '-' + str(round(val_score,4))+'-'+ str(('%s-%s-%s' % (now.hour, now.minute, now.second)) + weight_file_name))
                    util.remove_legacyModels(args.model_savepath) # Delete useless checkpoints.
            else:
                if val_loss < best_loss:
                    best_loss = val_loss
                    train_result['best_epoch'] = epoch + 1
                    train_result['best_loss'] = round(best_loss, 5)
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'lr': util.get_lr(optimizer), },
                        args.checkname + '-epoch' + str(epoch) + '-' + str(round(val_score,4)) + '-'+ str(('%s-%s-%s' % (now.hour, now.minute, now.second)) + weight_file_name))
            elapsed = time.time() - start_time
            epoch_lr = util.get_lr(optimizer)
            print('\n')

            print("Epoch {} - train_loss: {:.4f}  val_loss: {:.4f}  cv_score: {:.4f}  lr: {:.6f}  time: {:.0f}s".format(
                    epoch + 1, train_loss, val_loss, val_score, epoch_lr, elapsed))
            self.writer.add_scalar('epoch/learning_rate', epoch_lr, epoch)

            # scheduler update
            if args.is_scheduler == True:
                scheduler.step(val_score)

        return train_result, lrs, score

def main(writer):
    # Set base parameters (dataset path, backbone name etc...)
    parser = argparse.ArgumentParser(description="This code is for testing various models.")
    parser.add_argument('--backbone', type=str, default='efficientnet-b2',
                        choices=['efficientnet-b0','efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3', 'efficientnet-b4'],
                        help='Set backbone name. be careful input_channels when using oct_resnet.')
    parser.add_argument('--dataset', type=str, default='KaKR3rd', choices=['KPopGirls', 'KPopBoys'],
                        help='Set dataset type.')
    parser.add_argument('--dataset_path', type=str, default='./../KPopGirls', help='Set base dataset path.')
    parser.add_argument('--workers', type=int, default=4, metavar='N', help='Set CPU threads for pytorch dataloader')
    parser.add_argument('--checkname', type=str, default=None,
                        help='Set the checkpoint name. if None, checkname will be set to current dataset+backbone+time.')
    parser.add_argument('--model_savepath', type=str, default=None, help='set directory for saving trained model.')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                        help='전체 루틴을 설정합니다. `train`에는 train과 validation이 포함되어 있으므로, F1 score 계산 시엔 test로 변경해야 합니다.')

    # Set hyper params for training network.
    parser.add_argument('--loss_type', type=str, default='ce', choices=['ce', 'focal'],
                        help='Set loss func type. `ce` is crossentropy, `focal` is focal entropy from DeeplabV3.')
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='Set max epoch. If None, max epoch will be set to current dataset`s max epochs.')
    parser.add_argument('--batch_size', type=int, default=None,
                        metavar='N', help='input batch size for training')
    parser.add_argument('--image_size', type=int, default=None, help='input image size for training')
    parser.add_argument('--ref_batch_size', type=int, default=None,
                        metavar='N', help='input batch size for retrieval (default: auto)')
    parser.add_argument('--class_num', type=int, default=None,
                        help='Set class number. If None, class_num will be set according to dataset`s class number.')
    parser.add_argument('--use_pretrained', type=bool, default=False) # ImageNet-1000 pre-trained model 사용여부(=finetuning 할지, 말지 여부)
    parser.add_argument('--feature_extract', type=bool, default=True) # 추가로 쌓은 레이어만 학습('True')할지, pretrained weight 전체를 재학습('False')할지 결정

    # Set optimizer params for training network.
    parser.add_argument('--lr', type=float, default=None,
                        help='Set starting learning rate. If None, lr will be set to current dataset`s lr.')
    parser.add_argument('--is_scheduler', type=bool, default=False)
    parser.add_argument('--lr_scheduler', type=str, default='CosineAnnealingWithRestartsLR',
                        choices=['StepLR', 'MultiStepLR', 'ReduceLROnPlateau', 'WarmupCosineSchedule',
                                 'CosineAnnealingWithRestartsLR', 'WarmupCosineWithHardRestartsSchedule', 'CosineAnnealingWithRestartsLR'],
                        help='Set lr scheduler mode: (default: WarmupCosineSchedule)')
    parser.add_argument('--optim', type=str, default='ADAM', choices=['SGD', 'ADAM', 'AdamW', 'RAdam'],
                        help='Set optimizer type. (default: RAdam)')
    parser.add_argument('--beta1', default=0.9, type=float, help='beta1 for adam')
    parser.add_argument('--beta2', default=0.999, type=float, help='beta2 for adam')
    parser.add_argument('--weight_decay', '--wd', default=0.000025, type=float, metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='Set momentum value for pytorch`s SGD optimizer. (default: 0.9)')

    # Set params for CUDA, seed and logging
    parser.add_argument('--gpu_ids', type=str, default='0', help='use which gpu to train, must be a comma-separated list of integers only (default=0)')
    parser.add_argument('--use_cuda', type=bool, default=None, help='wheher use CUDA (default=None)')

    args = parser.parse_args()
    print('cuDNN version : ', torch.backends.cudnn.version())
    print(os.listdir(args.dataset_path))

    # default settings for test models. (contains epochs, lr and class_num of dataset etc.)
    if args.epochs is None:
        epoches = {'KPopGirls': 300, 'KPopBoys': 300}
        args.epochs = epoches[args.dataset]
    if args.lr is None:
        # set initial learning rate.
        lrs = {'KPopGirls': 2.5e-4, 'KPopBoys': 2.5e-4} # AI-RUSH baseline : 2.5e-4
        args.lr = lrs[args.dataset]
    if args.class_num is None:
        class_nums = {'KPopGirls': 29, 'KPopBoys': None}
        args.class_num  = class_nums[args.dataset]
    if args.image_size is None:
        image_sizes = {'KPopGirls': 224, 'KPopBoys': 224}
        args.image_size = image_sizes[args.dataset]
    if args.batch_size is None:
        batch_nums = {'KPopGirls': 64, 'KPopBoys': 64}
        args.batch_size = batch_nums[args.dataset]
    if args.ref_batch_size is None: # Retrieval 단계에서 'ref_batch_nums'장 단위로 query image와 유사도를 비교합니다.
        ref_batch_nums = {'KPopGirls': 50, 'KPopBoys': 50}
        args.ref_batch_size = ref_batch_nums[args.dataset]
    if args.model_savepath or args.checkname is None:
        now = datetime.now()
        checkpoint_name = str(args.dataset) + '-' + str(args.backbone) +'_' + str(('%s-%s-%s' % (now.year, now.month, now.day)))
        args.model_savepath = './trained_models/' + checkpoint_name
        args.checkname = args.model_savepath + '/' + checkpoint_name
    if args.use_cuda is None:
        use_cuda = cuda.is_available()
        args.use_cuda = use_cuda
    print(args)

    # 학습된 모델이 저장될 디렉토리 존재여부 확인
    if not (os.path.isdir(args.model_savepath)):
        os.makedirs(args.model_savepath)
        print("New directory '" + str(args.model_savepath) + "' has been created for saving trained models.")

    # Define trainer. (trainer includes dataloader, model, optimizer etc...)
    Trainer(args, writer)

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    writer = SummaryWriter()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 0=2080ti, 1=RTX TITAN in Deeperence server

    SEED = 2020
    util.fix_seed_everything(SEED) # 하이퍼파라미터 테스트를 위해 모든 시드 고정
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    main(writer)