# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch optimization for BERT model."""

import logging, math
from torch import optim
from torch.optim.lr_scheduler import LambdaLR

logger = logging.getLogger(__name__)

class ConstantLRSchedule(LambdaLR):
    """ Constant learning rate schedule.
    """
    def __init__(self, optimizer, last_epoch=-1):
        super(ConstantLRSchedule, self).__init__(optimizer, lambda _: 1.0, last_epoch=last_epoch)

class WarmupConstantSchedule(LambdaLR):
    """ Linear warmup and then constant.
        Linearly increases learning rate schedule from 0 to 1 over `warmup_steps` training steps.
        Keeps learning rate schedule equal to 1. after warmup_steps.
    """
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super(WarmupConstantSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        return 1.

class WarmupLinearSchedule(LambdaLR):
    """ Linear warmup and then linear decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Linearly decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps.
    """

    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))

class WarmupCosineSchedule(LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """
    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))

class WarmupCosineWithHardRestartsSchedule(LambdaLR):
    """ Linear warmup and then cosine cycles with hard restarts.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        If `cycles` (default=1.) is different from default, learning rate follows `cycles` times a cosine decaying
        learning rate (with hard restarts).
    """

    def __init__(self, optimizer, warmup_steps, t_total, cycles=1., last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineWithHardRestartsSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        if progress >= 1.0:
            return 0.0
        return max(0.0, 0.5 * (1. + math.cos(math.pi * ((float(self.cycles) * progress) % 1.0))))

def defineLRScheduler(args, optimizer, length_train_dataloader):
    warmup_steps = int(length_train_dataloader * 0.15)

    # 파이토치 기본제공 learning rate scheduler
    if args.lr_scheduler.lower() == 'steplr':
        scheduler = optim.plr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    elif args.lr_scheduler.lower() == 'multisteplr':
        '''
        - milestones : 해당 epoch마다 lr을 *gamma배만큼 decay 
        - gamma : Multiplicative factor of learning rate decay. Default: 0.1.
        '''
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[2, 5, 10, 15, 30], gamma=0.5)
    elif args.lr_scheduler.lower() == 'reducelronplateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.1)
    elif args.lr_scheduler.lower() == 'cosineschedule':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=10, eta_min=0)

    # 파이토치 기본제공 learning rate의 수정버전
    elif args.lr_scheduler.lower() == 'warmupcosineschedule':
        # warmup_steps동안 0~1까지 warmup 후 1~0까지 t_total-warmup_steps의 step동안 cosine을 그리며 decay
        scheduler = WarmupCosineSchedule(optimizer=optimizer, warmup_steps=warmup_steps, t_total=length_train_dataloader, cycles=0.5, last_epoch=-1)
        print('Warm-up during 0 to ', warmup_steps, 'steps, Cosine decay during ', warmup_steps, ' to ', length_train_dataloader, 'steps.')
    elif args.scheduler == 'WarmupCosineWithHardRestartsSchedule':
        scheduler = WarmupCosineWithHardRestartsSchedule(optimizer=optimizer, warmup_steps=warmup_steps, t_total=length_train_dataloader, cycles=1.0, last_epoch=-1)
    elif args.scheduler == 'WarmupLinearSchedule': # Constant learning rate scheduling.
        scheduler = ConstantLRSchedule(optimizer, last_epoch=-1)
    else:
        print("Invalid args input.")
        raise ValueError('Invalid scheduler')

    print('optimizer : ', type(optimizer), ' lr scheduler : ', type(scheduler))
    return scheduler