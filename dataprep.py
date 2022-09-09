#!/usr/bin/env python
# coding: utf-8



import os
import numpy as np
import cv2
import warnings
import math
import argparse

#os.chdir('C:/labels/train')
#os.listdir(os.getcwd())




#os.chdir('C:/labels')





def find_unique(master,sub):
    
    unique=[]
    #for i,m in enumerate(master):
    #    master[i]=master[i].split(".")[0]
    #master=master
    inst1=""
    for i,m in enumerate([m[:41] for m in master]):
        #print("==========")
        #print(m)
        is_unique=True
        for j,s in enumerate([s[:41] for s in sub]):
            #print("----------")
            #if
            #print("/t"+s)
            if m in s:
                is_unique=False
                break
        if is_unique:
            unique.append(master[i])
    return unique
#infer=find_unique(os.listdir(),os.listdir('C:/labels/m'))





def p1_to_deepglobe(path):
    """
    Given path to a phaseone image
    Return list of 1024x1024 crops
    """
    #path="C:/ausm/phaseone/labels/IMG_MM000XXX_2021-05-12_18-10-35.997_6661.tif"
    img=cv2.imread(path)
    S=0.1442047376 # scale factor to convert phaseone GSD to DeepGlobe GSD 
    img=cv2.resize(img,None,fx=S,fy=S)
    return [img[:1024,:1024],img[:1024,-1024:],img[-1024:,:1024],img[-1024:,-1024:]]

def resize_crop_train_test(inpath,outpath):
    """
    Given inpath to directory with test/train folder system
    Return test/train folders in outpath directory
    """
    trainmode=False
    try:
        os.mkdir(outpath)
    except:
        pass
    #files=os.listdir(path)
    #print(files)
    #files=list(filter(lambda x: x.endswith('.tif') or x.endswith('.tiff'),files))
    #files=list(filter(lambda x: x.endswith('.tiff'),files))
    for d in ['test','train']:
        try:
            os.mkdir(os.path.join(outpath,d))
        except:
            pass
        
        path=os.path.join(inpath,d)
        files=os.listdir(path)
        count=0
        excl_count=0
        for file in files:
            (pre,sep,suf)=file.rpartition('.')
            #print(pre,sep,suf)
            if pre[-4:].isnumeric() and suf=='tif':
                #print(pre)
                img=p1_to_deepglobe(os.path.join(path,file))
                lab=p1_to_deepglobe(os.path.join(path,pre+'_road.tif'))
                count+=1
                for i,r in enumerate(lab):
                    if np.sum(r)>0 or not trainmode:
                        #print(os.path.join(path,'resize','RSZ_'+pre+'_'+str(i)+sep+suf))
                        cv2.imwrite(os.path.join(outpath,d,'RSZ_'+pre+'_'+str(i)+'.tif'),img[i]) 
                        cv2.imwrite(os.path.join(outpath,d,'RSZ_'+pre+'_'+str(i)+'.tiff'),r)
                    else:
                        excl_count+=1
                        print(pre+'_'+str(i),f"\n{excl_count} label(s) with no roads detected and omitted")
        print(f"{excl_count} resized crops omitted")

def resize_crop_infer(inpath,outpath):
    """resize_crop_infer(inpath,outpath)
    Resize PhaseOne images in inpath to 0.5m GSD of DeepGlobe then slices out 1024 x 1024 crop out of corners
    Input: inpath - path to PhaseOne images
    Output: outpath - path to folder to output crops
    """
    files=os.listdir(inpath)

    for file in files:
        (pre,sep,suf)=file.rpartition('.')
        if suf=='tif':
            print(pre)
            imgs=p1_to_deepglobe(os.path.join(inpath,file))
            for i,img in enumerate(imgs):
                cv2.imwrite(os.path.join(outpath,'RSZ_'+pre+'_'+str(i)+'.tif'),img)

def merge_masks(inpath,outpath):
    """merge_masks(inpath):
    
        Merges 4 1024x1024 fuzzy masks into 10652x14204 binary masks from inpath to outpath
        
    """
    FUZZY=False
    EXT = 'fuzzy' if FUZZY else 'mask'
    #S=1/0.1442047376 # scale factor to convert phaseone GSD to DeepGlobe GSD 
    S=6.934584929 # scale factor to convert DeepGlobe GSD to phaseone GSD 
    
    overlap_arr=np.zeros([1536,2048])
    overlap_arr[:1024,:1024]+=1
    overlap_arr[:1024,-1024:]+=1
    overlap_arr[-1024:,:1024]+=1
    overlap_arr[-1024:,-1024:]+=1

    def myTile(file,tile):
        index=int(file.split('_')[-2])
        img=cv2.imread(os.path.join(inpath,file),0)
        if index == 0:
            tile[:1024,:1024]+=img
        elif index==1:
            tile[:1024,-1024:]+=img
        elif index==2:
            tile[-1024:,:1024]+=img
        elif index==3:
            tile[-1024:,-1024:]+=img
        return tile
    #try:
    #    os.mkdir(os.path.join(inpath,'merge'))
    #except:
    #    warnings.warn('/merge folder exists')
        
    isFirst=True
    img_num=0
    files=os.listdir(inpath)
    for file in files:
        if '.png' in file:
            img_num+=1
    print(img_num)

    

    count=0
    for i,file in enumerate(files):
        if '.png' in file:
            count+=1
            print(count)
            if isFirst:

                tile=np.zeros([1536,2048])
                curr_file=file.rsplit('_',2)[0]
                parent_file=curr_file
                tile=myTile(file,tile)
                isFirst=False
            else:
                curr_file=file.rsplit('_',2)[0]
                if curr_file == parent_file:
                    tile=myTile(file,tile)
                else:
                    tile=np.divide(tile,overlap_arr)                    
                    tile=cv2.resize(tile,(14204,10652),interpolation=cv2.INTER_LINEAR)
                    
                    if not FUZZY:
                        tile[tile>=128]=255
                        tile[tile<128]=0
                        
                    cv2.imwrite(os.path.join(outpath,parent_file.split('_',1)[1]+'_'+EXT+'.png'),
                                tile.astype(np.uint8))
                    
                    tile=np.zeros([1536,2048])
                    tile=myTile(file,tile)
                    
                    parent_file=curr_file
                    
                if count == img_num:
                    tile=np.divide(tile,overlap_arr)              
                    tile=cv2.resize(tile,(14204,10652),interpolation=cv2.INTER_LINEAR)
                    
                    if not FUZZY:                        
                        tile[tile>=128]=255
                        tile[tile<128]=0
                        
                    cv2.imwrite(os.path.join(outpath,parent_file.split('_',1)[1]+'_'+EXT+'.png'),
                                tile.astype(np.uint8))
        #break
                
#merge_masks('C:/labels/out/first02_merge','C:/labels/out/finetune')



def crops_to_phaseone(path):
    #src="Something unique"
    #crops = os.listdir(path)
    print("=== START ===")
    isFirst = True
    isRoot = True
    #print(path.split('/')[0])
    for (root, dirs, files) in os.walk(path, topdown=True):
        #print(dirs[0])
        out=[]
        if isRoot:
            isRoot=False
            continue
        if files:
            #print('Files detected in '+root)
            out=cv2.imread(os.path.join(root,files[0]))
        else: 
            continue
        for i, crop in enumerate(files[1:]):
            #pass
            #print(crop)
            out=out+cv2.imread(os.path.join(root,crop))
        outname=os.path.basename(os.path.normpath(root)).rpartition('.')[0]
        try:
            os.mkdir(os.path.join(root,os.pardir,os.pardir,'merge'))
        except:
            pass
        print(outname+".tiff")
        #print(os.listdir(os.path.join(root,os.pardir,os.pardir,'merge')))
        cv2.imwrite(os.path.join(root,os.pardir,os.pardir,'merge',outname+".tiff"), out.astype(np.uint8))

    print("=== FINISH ===")
    print('Results in ',os.path.abspath(os.path.join(path,os.pardir,'merge')))
parser = argparse.ArgumentParser()
parser.add_argument("inpath",help="path to masks_instances folder")

args = parser.parse_args()
crops_to_phaseone(args.inpath)