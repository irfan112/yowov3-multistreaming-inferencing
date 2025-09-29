import torch
import torch.utils.data as data
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as FT
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt
import time
import xml.etree.ElementTree as ET
import os
import cv2
import random
import sys
import glob

import json
from datetime import datetime

from math import sqrt
from utils.gradflow_check import plot_grad_flow
from utils.EMA import EMA
import logging
from utils.build_config import build_config
from cus_datasets.ucf.load_data import UCF_dataset
from cus_datasets.collate_fn import collate_fn
from cus_datasets.build_dataset import build_dataset
from model.TSN.YOWOv3 import build_yowov3 
from utils.loss import build_loss
from utils.warmup_lr import LinearWarmup
import shutil
from utils.flops import get_info

class EpochLogger:
    def __init__(self, save_folder):
        self.save_folder = save_folder
        self.epoch_logs = []
        self.log_file = os.path.join(save_folder, "epoch_training_log.json")
        self.txt_log_file = os.path.join(save_folder, "epoch_training_log.txt")
        
        # Initialize log files
        with open(self.txt_log_file, "w") as f:
            f.write(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n")
    
    def log_epoch(self, epoch_info):
        self.epoch_logs.append(epoch_info)
        # Save to JSON
        with open(self.log_file, "w") as f:
            json.dump(self.epoch_logs, f, indent=2)
        # Save to text
        with open(self.txt_log_file, "a") as f:
            f.write(f"Epoch {epoch_info['epoch']:3d} | "
                   f"Avg Loss: {epoch_info['avg_loss']:.6f} | "
                   f"LR: {epoch_info['learning_rate']:.2e} | "
                   f"Time: {epoch_info['epoch_time']:.2f}s\n")


def train_model(config):

    # Save config file
    #######################################################
    source_file = config['config_path']
    destination_file = os.path.join(config['save_folder'], 'config.yaml')
    shutil.copyfile(source_file, destination_file)
    #######################################################
    
    # create dataloader, model, criterion
    ####################################################
    dataset = build_dataset(config, phase='train')
    
    dataloader = data.DataLoader(dataset, config['batch_size'], True, collate_fn=collate_fn
                                 , num_workers=config['num_workers'], pin_memory=True)
    
    model = build_yowov3(config)
    get_info(config, model)
    model.to("cuda")
    model.train()
    
    criterion = build_loss(model, config)
    #####################################################

    #optimizer  = optim.AdamW(params=model.parameters(), lr= config['lr'], weight_decay=config['weight_decay'])

    g = [], [], []  # optimizer parameter groups
    bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
    for v in model.modules():
        for p_name, p in v.named_parameters(recurse=0):
            if p_name == "bias":  # bias (no decay)
                g[2].append(p)
            elif p_name == "weight" and isinstance(v, bn):  # weight (no decay)
                g[1].append(p)
            else:
                g[0].append(p)  # weight (with decay)

    optimizer = torch.optim.AdamW(g[0], lr=config['lr'], weight_decay=config['weight_decay'])
    optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})  
    optimizer.add_param_group({"params": g[2], "weight_decay": 0.0}) 
    
    warmup_lr  = LinearWarmup(config)

    adjustlr_schedule = config['adjustlr_schedule']
    acc_grad          = config['acc_grad'] 
    max_epoch         = config['max_epoch'] 
    lr_decay          = config['lr_decay']
    save_folder       = config['save_folder']

    logger = EpochLogger(config['save_folder'])
    
    torch.backends.cudnn.benchmark = True
    cur_epoch = 1
    # loss_acc = 0.0
    ema = EMA(model)
    training_start_time = time.time()

    while(cur_epoch <= max_epoch):
        epoch_start_time = time.time()  # ADD THIS LINE
        epoch_losses = []
        cnt_pram_update = 0
        loss_acc = 0.0


        for iteration, (batch_clip, batch_bboxes, batch_labels) in enumerate(dataloader): 

            batch_size   = batch_clip.shape[0]
            batch_clip   = batch_clip.to("cuda")
            for idx in range(batch_size):
                batch_bboxes[idx]       = batch_bboxes[idx].to("cuda")
                batch_labels[idx]       = batch_labels[idx].to("cuda")

            outputs = model(batch_clip)

            targets = []
            for i, (bboxes, labels) in enumerate(zip(batch_bboxes, batch_labels)):
                nbox = bboxes.shape[0]
                nclass = labels.shape[1]
                target = torch.Tensor(nbox, 5 + nclass)
                target[:, 0] = i
                target[:, 1:5] = bboxes
                target[:, 5:] = labels
                targets.append(target)

            targets = torch.cat(targets, dim=0)

            loss = criterion(outputs, targets) / acc_grad
            loss_acc += loss.item()
            loss.backward()
            #plot_grad_flow(model.named_parameters()) #model too large, can't see anything!
            #plt.show()

            if (iteration + 1) % acc_grad == 0:
                cnt_pram_update = cnt_pram_update + 1
                if cur_epoch == 1:
                    warmup_lr(optimizer, cnt_pram_update)
                nn.utils.clip_grad_value_(model.parameters(), clip_value=2.0)
                optimizer.step()
                optimizer.zero_grad()
                ema.update(model)

                print("epoch : {}, update : {}, loss = {}".format(cur_epoch,  cnt_pram_update, loss_acc), flush=True)
                
                # REPLACE the existing file write with this:
                with open(os.path.join(config['save_folder'], "logging.txt"), "a") as f:  # Changed "w" to "a"
                    f.write("epoch : {}, update : {}, loss = {}\n".format(cur_epoch,  cnt_pram_update, loss_acc))

                epoch_losses.append(loss_acc)  # ADD THIS LINE
                loss_acc = 0.0

        current_lr = optimizer.param_groups[0]['lr']
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        total_time = epoch_end_time - training_start_time
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        
        # Create epoch info
        epoch_info = {
            "epoch": cur_epoch,
            "avg_loss": avg_loss,
            "min_loss": np.min(epoch_losses) if epoch_losses else 0.0,
            "max_loss": np.max(epoch_losses) if epoch_losses else 0.0,
            "learning_rate": current_lr,
            "epoch_time": epoch_time,
            "total_time": total_time,
            "num_updates": cnt_pram_update,
            "timestamp": datetime.now().isoformat()
        }


        if cur_epoch in adjustlr_schedule:
            for param_group in optimizer.param_groups: 
                param_group['lr'] *= lr_decay
            epoch_info["lr_adjusted"] = True  # ADD THIS LINE
        else:
            epoch_info["lr_adjusted"] = False  # ADD THIS LINE
        
        #          model.state_dict()
        save_path_ema = os.path.join(save_folder, "ema_epoch_" + str(cur_epoch) + ".pth")
        torch.save(ema.ema.state_dict(), save_path_ema)

        save_path     = os.path.join(save_folder, "epoch_"     + str(cur_epoch) + ".pth")
        torch.save(model.state_dict(), save_path)
        save_path = os.path.join(save_folder, "epoch_" + str(cur_epoch) + ".pth")
        torch.save(model.state_dict(), save_path)

        
        logger.log_epoch(epoch_info)
        print(f"Epoch {cur_epoch} completed | Avg Loss: {avg_loss:.6f} | Time: {epoch_time:.2f}s", flush=True)

        print("Saved model at epoch : {}".format(cur_epoch), flush=True)
        cur_epoch += 1

        #log_path = '/home/manh/Projects/YOLO2Stream/training.log'
        #map50, mean_ap = call_eval(save_path)
        #logging.basicConfig(filename=log_path, level=logging.INFO)
        #logging.info('mAP 0.5 : {}, mAP : {}'.format(map50, mean_ap))

        cur_epoch += 1

if __name__ == "__main__":
    config = build_config()
    train_model(config)   