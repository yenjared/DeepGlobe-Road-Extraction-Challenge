import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V

import cv2
import os
import numpy as np

from time import time

from networks.unet import Unet
from networks.dunet import Dunet
from networks.dinknet import LinkNet34, DinkNet34, DinkNet50, DinkNet101, DinkNet34_less_pool
from framework import MyFrame
from loss import dice_bce_loss
from data import ImageFolder

"""
TODO: Tune learning rate, line 32ish
"""

SHAPE = (1024,1024)
ROOT = 'dataset/train/'
imagelist = filter(lambda x: x[-3:].find('tif')!=-1, os.listdir(ROOT)) # training image list
trainlist = list(map(lambda x: x.rpartition('.')[0], imagelist)) # label prefix list
print(len(trainlist))
#NAME = 'test03_dink34'
NAME = 'postencode01_dink34'
#NAME = 'log01_dink34'
BATCHSIZE_PER_CARD = 4

# MyFrame from framework.py
# uses D-LinkNet34, dice binary cross entropy loss function and 2e-4 learning rate
solver = MyFrame(DinkNet34, dice_bce_loss, 2e-3) # TUNE LEARNING RATE
batchsize = torch.cuda.device_count() * BATCHSIZE_PER_CARD

# torch.utils.data.Dataset abstract class instance, returns image and label, in data.py
dataset = ImageFolder(trainlist, ROOT)

data_loader = torch.utils.data.DataLoader( # combines dataset and sampler
    dataset,
    batch_size=batchsize,
    shuffle=True, # shuffles data every epoch
    num_workers=4)

file=mylog = open('logs/'+NAME+'.log','w')
tic = time()
no_optim = 0
#total_epoch = 10
total_epoch = 300

train_epoch_best_loss = 100.
for epoch in range(1, total_epoch + 1):

    # iter returns a shuffled dataset for every epoch
    data_loader_iter = iter(data_loader)

    train_epoch_loss = 0
    for img, mask in data_loader_iter:
        solver.set_input(img, mask)
        train_loss = solver.optimize()
        train_epoch_loss += train_loss
    train_epoch_loss /= len(data_loader_iter)
    print('********',file=mylog)
    print('epoch:',epoch,'    time:',int(time()-tic),file=mylog)
    print('train_loss:',train_epoch_loss,file=mylog)
    print('SHAPE:',SHAPE,file=mylog)
    print('********')
    print('epoch:',epoch,'    time:',int(time()-tic))
    print('train_loss:',train_epoch_loss)
    print('SHAPE:',SHAPE)

    if train_epoch_loss >= train_epoch_best_loss:
        no_optim += 1
    else:
        no_optim = 0
        train_epoch_best_loss = train_epoch_loss
        solver.save('weights/'+NAME+'.th')
    if no_optim > 6:
        print('early stop at %d epoch' % epoch,file=mylog)
        print('early stop at %d epoch' % epoch)
        break
    if no_optim > 3:
        if solver.old_lr < 5e-7:
            break
        solver.load('weights/'+NAME+'.th')
        solver.update_lr(5.0,mylog,factor = True)
    file=mylog.flush()

print('Finish!',file=mylog)
print('Finish!')
file=mylog.close()
