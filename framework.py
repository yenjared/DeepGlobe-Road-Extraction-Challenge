import torch
import torch.nn as nn
from torch.autograd import Variable as V

import cv2
import numpy as np

from torchvision import models
import torch.nn.functional as F

from functools import partial
from networks.dinknet import Dblock, DecoderBlock

nonlinearity = partial(F.relu,inplace=True)

class MyFrame():
    def __init__(self, net, loss, lr=2e-4, evalmode = False):
        self.net = net().cuda() # aka nn.Module Dinknet34 in dinknet.py

        #""" Start of Fine-tuning block
        self.net.load_state_dict(torch.load('weights/log01_dink34.th'),strict=False)
        
        
        for param in self.net.parameters():
            param.requires_grad = False
            
        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        #self.net.firstconv = resnet.conv1
        #self.net.firstbn = resnet.bn1
        #self.net.firstrelu = resnet.relu
        #self.net.firstmaxpool = resnet.maxpool

        self.net.encoder1 = resnet.layer1
        self.net.encoder2 = resnet.layer2
        self.net.encoder3 = resnet.layer3
        self.net.encoder4 = resnet.layer4
        
        self.net.dblock = Dblock(512)

        self.net.decoder4 = DecoderBlock(filters[3], filters[2])
        self.net.decoder3 = DecoderBlock(filters[2], filters[1])
        self.net.decoder2 = DecoderBlock(filters[1], filters[0])
        self.net.decoder1 = DecoderBlock(filters[0], filters[0])
        
        self.net.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.net.finalrelu1 = nonlinearity
        self.net.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.net.finalrelu2 = nonlinearity
        self.net.finalconv3 = nn.Conv2d(32, 1, 3, padding=1)
        
        print('=== Unfrozen layers ===')
        for name, param in self.net.named_parameters():
            if param.requires_grad:
                print(name)
        print('=======================')
        #""" End of Fine-Tuning Block
        self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))
        self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=lr)
        #self.optimizer = torch.optim.RMSprop(params=self.net.parameters(), lr=lr)
        self.loss = loss()
        self.old_lr = lr
        if evalmode:
            for i in self.net.modules():
                if isinstance(i, nn.BatchNorm2d):
                    i.eval()

    def set_input(self, img_batch, mask_batch=None, img_id=None):
        self.img = img_batch
        self.mask = mask_batch
        self.img_id = img_id

    def test_one_img(self, img):
        pred = self.net.forward(img)

        pred[pred>0.5] = 1
        pred[pred<=0.5] = 0

        mask = pred.squeeze().cpu().data.numpy()
        return mask

    def test_batch(self):
        self.forward(volatile=True)
        mask =  self.net.forward(self.img).cpu().data.numpy().squeeze(1)
        mask[mask>0.5] = 1
        mask[mask<=0.5] = 0

        return mask, self.img_id

    def test_one_img_from_path(self, path):
        img = cv2.imread(path)
        img = np.array(img, np.float32)/255.0 * 3.2 - 1.6
        img = V(torch.Tensor(img).cuda())

        mask = self.net.forward(img).squeeze().cpu().data.numpy()#.squeeze(1)
        mask[mask>0.5] = 1
        mask[mask<=0.5] = 0

        return mask

    def forward(self, volatile=False):
        self.img = V(self.img.cuda(), volatile=volatile)
        if self.mask is not None:
            self.mask = V(self.mask.cuda(), volatile=volatile)

    def optimize(self):
        self.forward()
        self.optimizer.zero_grad()
        pred = self.net.forward(self.img)
        loss = self.loss(self.mask, pred)
        loss.backward()
        self.optimizer.step()
        return loss.data.item()
        #return loss.data[0]

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path))

    def update_lr(self, new_lr, mylog, factor=False):
        if factor:
            new_lr = self.old_lr / new_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

        print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
        print('update learning rate: %f -> %f' % (self.old_lr, new_lr),mylog)
        self.old_lr = new_lr
