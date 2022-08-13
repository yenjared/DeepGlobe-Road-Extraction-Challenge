import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

from time import time

from networks.unet import Unet
from networks.dunet import Dunet
from networks.dinknet import LinkNet34, DinkNet34, DinkNet50, DinkNet101, DinkNet34_less_pool

from sklearn.metrics import jaccard_score, precision_recall_fscore_support

BATCHSIZE_PER_CARD = 8

class TTAFrame():
    def __init__(self, net):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = net().to(device)
        self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))
        
    def test_one_img_from_path(self, path, evalmode = True):
        if evalmode:
            self.net.eval()
        #batchsize = torch.cuda.device_count() * BATCHSIZE_PER_CARD
        batchsize = BATCHSIZE_PER_CARD
        #batchsize = BATCHSIZE_PER_CARD if torch.cuda.is_available() else BATCHSIZE_PER_CARD * torch.cuda.device_count()
        
        if batchsize >= 8:
            return self.test_one_img_from_path_1(path)
        elif batchsize >= 4:
            return self.test_one_img_from_path_2(path)
        elif batchsize >= 2:
            return self.test_one_img_from_path_4(path)

    def test_one_img_from_path_8(self, path):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        img = cv2.imread(path)#.transpose(2,0,1)[None]
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None],img90[None]])
        img2 = np.array(img1)[:,::-1]
        img3 = np.array(img1)[:,:,::-1]
        img4 = np.array(img2)[:,:,::-1]
        
        img1 = img1.transpose(0,3,1,2)
        img2 = img2.transpose(0,3,1,2)
        img3 = img3.transpose(0,3,1,2)
        img4 = img4.transpose(0,3,1,2)
        
        img1 = V(torch.Tensor(np.array(img1, np.float32)/255.0 * 3.2 -1.6).to(device))
        img2 = V(torch.Tensor(np.array(img2, np.float32)/255.0 * 3.2 -1.6).to(device))
        img3 = V(torch.Tensor(np.array(img3, np.float32)/255.0 * 3.2 -1.6).to(device))
        img4 = V(torch.Tensor(np.array(img4, np.float32)/255.0 * 3.2 -1.6).to(device))
        
        maska = self.net.forward(img1).squeeze().cpu().data.numpy()
        maskb = self.net.forward(img2).squeeze().cpu().data.numpy()
        maskc = self.net.forward(img3).squeeze().cpu().data.numpy()
        maskd = self.net.forward(img4).squeeze().cpu().data.numpy()
        
        mask1 = maska + maskb[:,::-1] + maskc[:,:,::-1] + maskd[:,::-1,::-1]
        mask2 = mask1[0] + np.rot90(mask1[1])[::-1,::-1]
        
        return mask2

    def test_one_img_from_path_4(self, path):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        img = cv2.imread(path)#.transpose(2,0,1)[None]
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None],img90[None]])
        img2 = np.array(img1)[:,::-1]
        img3 = np.array(img1)[:,:,::-1]
        img4 = np.array(img2)[:,:,::-1]
        
        img1 = img1.transpose(0,3,1,2)
        img2 = img2.transpose(0,3,1,2)
        img3 = img3.transpose(0,3,1,2)
        img4 = img4.transpose(0,3,1,2)
        
        img1 = V(torch.Tensor(np.array(img1, np.float32)/255.0 * 3.2 -1.6).to(device))
        img2 = V(torch.Tensor(np.array(img2, np.float32)/255.0 * 3.2 -1.6).to(device))
        img3 = V(torch.Tensor(np.array(img3, np.float32)/255.0 * 3.2 -1.6).to(device))
        img4 = V(torch.Tensor(np.array(img4, np.float32)/255.0 * 3.2 -1.6).to(device))
        
        maska = self.net.forward(img1).squeeze().cpu().data.numpy()
        maskb = self.net.forward(img2).squeeze().cpu().data.numpy()
        maskc = self.net.forward(img3).squeeze().cpu().data.numpy()
        maskd = self.net.forward(img4).squeeze().cpu().data.numpy()
        
        mask1 = maska + maskb[:,::-1] + maskc[:,:,::-1] + maskd[:,::-1,::-1]
        mask2 = mask1[0] + np.rot90(mask1[1])[::-1,::-1]
        
        return mask2
    
    def test_one_img_from_path_2(self, path):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        img = cv2.imread(path)#.transpose(2,0,1)[None]
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None],img90[None]])
        img2 = np.array(img1)[:,::-1]
        img3 = np.concatenate([img1,img2])
        img4 = np.array(img3)[:,:,::-1]
        img5 = img3.transpose(0,3,1,2)
        img5 = np.array(img5, np.float32)/255.0 * 3.2 -1.6
        img5 = V(torch.Tensor(img5).to(device))
        img6 = img4.transpose(0,3,1,2)
        img6 = np.array(img6, np.float32)/255.0 * 3.2 -1.6
        img6 = V(torch.Tensor(img6).to(device))
        
        maska = self.net.forward(img5).squeeze().cpu().data.numpy()#.squeeze(1)
        maskb = self.net.forward(img6).squeeze().cpu().data.numpy()
        
        mask1 = maska + maskb[:,:,::-1]
        mask2 = mask1[:2] + mask1[2:,::-1]
        mask3 = mask2[0] + np.rot90(mask2[1])[::-1,::-1]
        
        return mask3
    
    def test_one_img_from_path_1(self, path):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        img = cv2.imread(path)#.transpose(2,0,1)[None]

        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None],img90[None]])
        img2 = np.array(img1)[:,::-1]
        img3 = np.concatenate([img1,img2])
        img4 = np.array(img3)[:,:,::-1]
        img5 = np.concatenate([img3,img4]).transpose(0,3,1,2)
        img5 = np.array(img5, np.float32)/255.0 * 3.2 -1.6
        img5 = V(torch.Tensor(img5).to(device))
        
        mask = self.net.forward(img5).squeeze().cpu().data.numpy()#.squeeze(1)
        mask1 = mask[:4] + mask[4:,:,::-1]
        mask2 = mask1[:2] + mask1[2:,::-1]
        mask3 = mask2[0] + np.rot90(mask2[1])[::-1,::-1]
        
        return mask3

    def load(self, path):
        self.net.load_state_dict(torch.load(path,map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))
        
source = 'dataset/test/'
#source = 'dataset/valid/'
val = os.listdir(source)
solver = TTAFrame(DinkNet34)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
       
solver.load('weights/log01_dink34.th')
tic = time()
target = 'submits/log01_dink34/'

try:
  os.mkdir(target)
except:
  pass

lab=list(filter(lambda x: x.endswith('tif'),val))

iou=[]
precision=[]
recall=[]
f1score=[]

for i,name in enumerate(lab):
    if i%10 == 0:
        print(i/10, '    ','%.2f'%(time()-tic))

    gt=list(filter(lambda x: x.endswith('.tiff') 
                         and x.find(name.rpartition('.')[0])!=-1,val))[0]
    gt=cv2.imread(source+gt,0)
    gt[gt>128] = 255
    gt[gt<=128] = 0

    mask = solver.test_one_img_from_path(source+name)
    mask[mask>4.0] = 255
    mask[mask<=4.0] = 0
    
    print(type(gt))
    print(gt.shape)
    print(type(mask))
    print(mask.shape)

    iou_curr=jaccard_score(gt,mask,average='micro')
    prf=precision_recall_fscore_support(gt,mask,average='micro')

    iou.append(iou_curr)
    precision.append(prf[0])
    recall.append(prf[1])
    f1score.append(prf[2])

    print(source+name)
    mask = np.concatenate([mask[:,:,None],mask[:,:,None],mask[:,:,None]],axis=2)

    #print(source+name, '\n',iou_curr)
    #print(target+name.rsplit('.')+'mask.png')
    #target='/content/'
    #if isFirst
    #cv2.imwrite(target+name.rpartition('.')[0]+'_mask.png',mask.astype(np.uint8))
    break

with open(target+'performance_log.txt','a') as f:
  f.write("Baseline"+"\n")
  f.write('IoU,P,R,F\n')
  f.write('%s' % float('%.4g' % np.mean(np.array(iou)))+"," +
          '%s' % float('%.4g' % np.mean(np.array(precision))) + "," +
          '%s' % float('%.4g' % np.mean(np.array(recall)))+","+
          '%s' % float('%.4g' % np.mean(np.array(f1score)))+"\n")  