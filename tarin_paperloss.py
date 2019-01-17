import sys, os
import cv2
import torch
import argparse
import timeit
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import paramter
from torch.utils import data

from tqdm import tqdm

from models import get_model
from loaders import get_loader, get_data_path
from misc.metrics import runningScore, accuracy, mapk
from misc.utils import convert_state_dict, poly_lr_scheduler, AverageMeter

torch.backends.cudnn.benchmark = True

class Adptive_loss(nn.Module):
    def __init__(self):
        super(Adptive_loss, self).__init__()
        self.word_embedding = torch.FloatTensor(np.load('vector.npy'))
        self.word_embedding = self.word_embedding.cuda()

    def reduce_shaper(self, t):
        return torch.reshape(torch.sum(t, 1), [t.shape[0], 1])
    
    def convert_one_hot(self, y, C):
        return torch.eye(C)[y.reshape(-1)]

    def forward(self, outputs, labels):
    
        image_label = self.convert_one_hot(labels, args.batch_size)  # 512,340
        image_label = image_label.cuda()
        image_feature = outputs
        image_feature = image_feature.cuda()
        v_label = torch.mul(image_label.unsqueeze(2), self.word_embedding.cuda())  # 512,340,300
        ip_1 = torch.sum(torch.mul(image_feature.unsqueeze(1), v_label), 2)  # 512,340
        mod_1 = torch.sqrt(
            torch.mul(torch.sum(torch.pow(image_feature, 2), 1).unsqueeze(1),
                      torch.sum(torch.pow(v_label, 2), 2)))  # 512,340
        cos_B = torch.div(ip_1, mod_1)  # 512,340
        
        vi = torch.matmul(image_label.cuda(), self.word_embedding.cuda())  # 512,300
        ip_2 = torch.sum(torch.mul(image_feature, vi), 1)
        mod_2 = torch.sqrt(torch.mul(torch.sum(torch.pow(vi, 2), 1), torch.sum(torch.pow(image_feature, 2), 1)))
        cos_A = torch.div(ip_2, mod_2)  # 512

        v_label = torch.mul(image_label.unsqueeze(2), self.word_embedding.cuda())  # 512,340,300
        margin_ip1 = torch.sum(torch.mul(vi.unsqueeze(1), v_label), 2)  # 512,340
        margin_mod = torch.sqrt(
            torch.mul(torch.sum(torch.pow(vi, 2), 1).unsqueeze(1),
                      torch.sum(torch.pow(v_label, 2), 2)))  # 512,340
        margin_parm = torch.div(ip_1, mod_1)  # 512,340
        zeros = torch.zeros(args.batch_size, n_classes)

        loss1 = margin_parm + cos_B - 2 * cos_A.unsqueeze(1)
        loss1 = cos_B
        final_loss = torch.sum(loss1)

        return final_loss        
    
    

def train(args):
    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')

    # Setup Augmentations
    data_aug = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
                ])

    # Setup Dataloader
    data_loader = get_loader(args.dataset)
    data_path = get_data_path(args.dataset)
    t_loader = data_loader(data_path, is_transform=True, split='train', version='simplified', img_size=(args.img_rows, args.img_cols), augmentations=data_aug, train_fold_num=args.train_fold_num, num_train_folds=args.num_train_folds, seed=args.seed)
    v_loader = data_loader(data_path, is_transform=True, split='val', version='simplified', img_size=(args.img_rows, args.img_cols), num_val=args.num_val, seed=args.seed)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    n_classes = t_loader.n_classes
    trainloader = data.DataLoader(t_loader, batch_size=args.batch_size, num_workers=2, shuffle=True, pin_memory=True, drop_last=True)
    valloader = data.DataLoader(v_loader, batch_size=args.batch_size, num_workers=2, pin_memory=True)

    # Setup Metrics
    running_metrics = runningScore(n_classes)

    # Setup Model
    v_demision = 300
    model = get_model(args.arch, v_demision, use_cbam=args.use_cbam)
    model.cuda()

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.l_rate, weight_decay=args.weight_decay)
    if args.num_cycles > 0:
        len_trainloader = int(5e6) # 4960414
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_train_folds*len_trainloader//args.num_cycles, eta_min=args.l_rate*1e-1)
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 4, 6, 8], gamma=0.5)

    start_epoch = 0
    if args.resume is not None:                                         
        if os.path.isfile(args.resume):
            print("Loading model and optimizer from checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)

            model_dict = model.state_dict()
            if checkpoint.get('model_state', -1) == -1:
                model_dict.update(convert_state_dict(checkpoint, load_classifier=args.load_classifier))
            else:
                model_dict.update(convert_state_dict(checkpoint['model_state'], load_classifier=args.load_classifier))

                print("Loaded checkpoint '{}' (epoch {}, mapk {:.5f}, top1_acc {:7.3f}, top2_acc {:7.3f} top3_acc {:7.3f})"
                      .format(args.resume, checkpoint['epoch'], checkpoint['mapk'], checkpoint['top1_acc'], checkpoint['top2_acc'], checkpoint['top3_acc']))
            model.load_state_dict(model_dict)

            if checkpoint.get('optimizer_state', None) is not None:
                optimizer.load_state_dict(checkpoint['optimizer_state'])
                start_epoch = checkpoint['epoch']
        else:
            print("No checkpoint found at '{}'".format(args.resume)) 


    loss_sum = 0.0
    for epoch in range(start_epoch, args.n_epoch):
        start_train_time = timeit.default_timer()

        if args.num_cycles == 0:
            scheduler.step(epoch)

        model.train()
        optimizer.zero_grad()
        for i, (images, labels, recognized, _) in enumerate(trainloader):
            if args.num_cycles > 0:
                iter_num = i + epoch * len_trainloader
                scheduler.step(iter_num % (args.num_train_folds * len_trainloader // args.num_cycles)) # Cosine Annealing with Restarts

            images = images.cuda()
            labels = labels.cuda()

            outputs = model(images)
            a_loss = Adptive_loss().cuda()
            loss = a_loss(outputs, labels)
            loss = loss / float(args.iter_size) # Accumulated gradients
            loss_sum = loss_sum + loss

            loss.backward()

            if (i+1) % args.print_train_freq == 0:
                print("Epoch [%d/%d] Iter [%6d/%6d] Loss: %.4f" % (epoch+1, args.n_epoch, i+1, len(trainloader), loss_sum))

            if (i+1) % args.iter_size == 0 or i == len(trainloader) - 1:
                optimizer.step()
                optimizer.zero_grad()
                loss_sum = 0.0
        
        elapsed_train_time = timeit.default_timer() - start_train_time
        print('Training time (epoch {0:5d}): {1:10.5f} seconds'.format(epoch+1, elapsed_train_time))



if __name__ == '__main__':

    args = paramter.Arg()

    train(args)

