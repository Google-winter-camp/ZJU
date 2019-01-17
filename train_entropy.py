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
import paramters
from torch.utils import data
from tqdm import tqdm
from models import get_model
from loaders import get_loader, get_data_path
from misc.metrics import runningScore, accuracy, mapk
from misc.utils import convert_state_dict, poly_lr_scheduler, AverageMeter

torch.backends.cudnn.benchmark = True

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
    model = get_model(args.arch, n_classes, use_cbam=args.use_cbam)
    model.cuda()

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.l_rate, weight_decay=args.weight_decay)

    if args.num_cycles > 0:
        len_trainloader = int(5e6) # 4960414
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_train_folds*len_trainloader//args.num_cycles, eta_min=args.l_rate*1e-1)
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 4, 6, 8], gamma=0.5)
    
    loss_fn = F.cross_entropy

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
            recognized = recognized.cuda()

            outputs = model(images)

            loss = (loss_fn(outputs, labels.view(-1), ignore_index=t_loader.ignore_index, reduction='none') * recognized.view(-1)).mean()
            loss = loss / float(args.iter_size) # Accumulated gradients
            loss_sum = loss_sum + loss

            loss.backward()

            if (i+1) % args.print_train_freq == 0:
                print("Epoch [%d/%d] Iter [%6d/%6d] Loss: %.4f" % (epoch+1, args.n_epoch, i+1, len(trainloader), loss_sum))

            if (i+1) % args.iter_size == 0 or i == len(trainloader) - 1:
                optimizer.step()
                optimizer.zero_grad()
                loss_sum = 0.0

        mapk_val = AverageMeter()
        top1_acc_val = AverageMeter()
        top2_acc_val = AverageMeter()
        top3_acc_val = AverageMeter()
        mean_loss_val = AverageMeter()
        model.eval()
        with torch.no_grad():
            for i_val, (images_val, labels_val, recognized_val, _) in tqdm(enumerate(valloader)):
                images_val = images_val.cuda()
                labels_val = labels_val.cuda()
                recognized_val = recognized_val.cuda()

                outputs_val = model(images_val)

                loss_val = (loss_fn(outputs_val, labels_val.view(-1), ignore_index=v_loader.ignore_index, reduction='none') * recognized_val.view(-1)).mean()
                mean_loss_val.update(loss_val, n=images_val.size(0))

                _, pred = outputs_val.topk(k=3, dim=1, largest=True, sorted=True)
                running_metrics.update(labels_val, pred[:, 0])

                acc1, acc2, acc3 = accuracy(outputs_val, labels_val, topk=(1, 2, 3))
                top1_acc_val.update(acc1, n=images_val.size(0))
                top2_acc_val.update(acc2, n=images_val.size(0))
                top3_acc_val.update(acc3, n=images_val.size(0))

                mapk_v = mapk(labels_val, pred, k=3)
                mapk_val.update(mapk_v, n=images_val.size(0))

        print('Mean Average Precision (MAP) @ 3: {:.5f}'.format(mapk_val.avg))
        print('Top 3 accuracy: {:7.3f} / {:7.3f} / {:7.3f}'.format(top1_acc_val.avg, top2_acc_val.avg, top3_acc_val.avg))
        print('Mean val loss: {:.4f}'.format(mean_loss_val.avg))

        score, class_iou = running_metrics.get_scores()

        for k, v in score.items():
            print(k, v)


        state = {'epoch': epoch+1,
                 'model_state': model.state_dict(),
                 'optimizer_state': optimizer.state_dict(),
                 'mapk': mapk_val.avg,
                 'top1_acc': top1_acc_val.avg,
                 'top2_acc': top2_acc_val.avg,
                 'top3_acc': top3_acc_val.avg,}
        torch.save(state, "checkpoints/{}_{}_{}_{}x{}_{}-{}-{}_model.pth".format(args.arch, args.dataset, epoch+1, args.img_rows, args.img_cols, args.train_fold_num, args.num_train_folds, args.num_val))

        running_metrics.reset()
        mapk_val.reset()
        top1_acc_val.reset()
        top2_acc_val.reset()
        top3_acc_val.reset()
        mean_loss_val.reset()

        elapsed_train_time = timeit.default_timer() - start_train_time
        print('Training time (epoch {0:5d}): {1:10.5f} seconds'.format(epoch+1, elapsed_train_time))



if __name__ == '__main__':

    args = paramter.Arg()
    
    train(args)
