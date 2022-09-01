import sys
sys.path.append('../') #root path
from functools import partial
import os
import cv2
import datetime
import time
import argparse

import numpy as np
import tensorflow as tf

from data import *
from model import *

#Set variables
start = time.time()
time_now = datetime.datetime.now()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="dataset/train/", help="path to input image")
parser.add_argument("--init_lr", type=float, default=0.001, help="initial learning rate")
parser.add_argument("--min_lr", type=float, default=0.0001, help="minimun learning rate")
parser.add_argument("--batch_size", type=int, default=16, help="batch size for training")
parser.add_argument("--jpeg_quality", type=int, default=20, help="jpeg quality to be induced in the image")
parser.add_argument("--steps", type=int, default=2000, help="number of steps to be taken while training")
args = parser.parse_args()


# Generate folders to save checkpoints 
def generate_expname_automatically():
    name = "UNET_%s_%02d_%02d_%02d_%02d_%02d" % ("assignment",
            time_now.month, time_now.day, time_now.hour,
            time_now.minute, time_now.second)
    return name

expname  = generate_expname_automatically()
checkpoint_dir = expname ; check_folder("./__outputs/checkpoints/")
summary_dir = expname ; check_folder("./__outputs/summary/")

model = Model_Train(summary_dir, checkpoint_dir, args.init_lr, args.min_lr)

""" restore model """
if False:
    model.ckpt.restore(config.restore_file)

trainset_dispenser = Dataset_Dispenser(args.data_path,args.jpeg_quality, 48, args.batch_size)

count = 0
steps = 0

while steps <= args.steps: #manually stopping
    """ train """
    log, output = model.train_step(trainset_dispenser, log_interval= 100)
    if model.step.numpy() % 1 == 0:
        print("[train] step:{} elapse:{} {}".format(model.step.numpy(), time.time() - start, log))

        #visualization
        output_concat = np.concatenate([output[i] for i in range(len(output))], axis=1)[0]
        output_concat = cv2.resize(output_concat,(output_concat.shape[1]*3,output_concat.shape[0]*3))
        cv2.imwrite('outputs/image_'+str(count)+'_.png', output_concat[...,::-1])
        count+=1

    """ save model """
    if model.step.numpy() % 100 == 0:  save_path = model.save()
    model.step.assign_add(1)
    steps += 1
