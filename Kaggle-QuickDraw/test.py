import sys, os
import cv2
import torch
import argparse
import timeit
import random
import collections
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.backends import cudnn
from torch.utils import data

from tqdm import tqdm

from models import get_model
from loaders import get_loader, get_data_path
from misc.utils import convert_state_dict, flip, AverageMeter
from misc.metrics import runningScore, accuracy, mapk

cudnn.benchmark = True

def test(args):
    model_file_name = os.path.split(args.model_path)[1]
    model_name = model_file_name[:model_file_name.find('_')]

    # Setup Dataloader
    data_loader = get_loader(args.dataset)
    data_path = get_data_path(args.dataset)
    loader = data_loader(data_path, split=args.split, is_transform=True, img_size=(args.img_rows, args.img_cols), no_gt=args.no_gt, seed=args.seed)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    n_classes = loader.n_classes
    testloader = data.DataLoader(loader, batch_size=args.batch_size, num_workers=4, pin_memory=True)

    # Setup Model
    model = get_model(model_name, n_classes, use_cbam=args.use_cbam)
    model.cuda()

    checkpoint = torch.load(args.model_path)
    state = convert_state_dict(checkpoint['model_state'])
    model_dict = model.state_dict()
    model_dict.update(state)
    model.load_state_dict(model_dict)

    print("Loaded checkpoint '{}' (epoch {}, mapk {:.5f}, top1_acc {:7.3f}, top2_acc {:7.3f} top3_acc {:7.3f})"
          .format(args.model_path, checkpoint['epoch'], checkpoint['mapk'], checkpoint['top1_acc'], checkpoint['top2_acc'], checkpoint['top3_acc']))

    running_metrics = runningScore(n_classes)

    pred_dict = collections.OrderedDict()
    mapk = AverageMeter()
    model.eval()
    with torch.no_grad():
        for i, (images, labels, _, names) in tqdm(enumerate(testloader)):
            images = images.cuda()
            if args.tta:
                images_flip = flip(images, dim=3)

            outputs = model(images)
            if args.tta:
                outputs_flip = model(images_flip)

            prob = F.softmax(outputs, dim=1)
            if args.tta:
                prob_flip = F.softmax(outputs_flip, dim=1)
                prob = (prob + prob_flip) / 2.0

            _, pred = prob.topk(k=3, dim=1, largest=True, sorted=True)
            for k in range(images.size(0)):
                pred_dict[int(names[0][k])] = loader.encode_pred_name(pred[k, :])

            if not args.no_gt:
                running_metrics.update(labels, pred)

                mapk_val = mapk(labels, pred, k=3)
                mapk.update(mapk_val, n=images.size(0))


    if not args.no_gt:
        print('Mean Average Precision (MAP) @ 3: {:.5f}'.format(mapk.avg))

        score, class_iou = running_metrics.get_scores()

        for k, v in score.items():
            print(k, v)

        #for i in range(n_classes):
        #    print(i, class_iou[i])

        running_metrics.reset()
        mapk.reset()

    # Create submission
    sub = pd.DataFrame.from_dict(pred_dict, orient='index')
    sub.index.names = ['key_id']
    sub.columns = ['word']
    sub.to_csv('{}_{}x{}.csv'.format(args.split, args.img_rows, args.img_cols))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--model_path', nargs='?', type=str, default='MobileNetV2_quickdraw_50_128x128_9-10-300_model.pth',
                        help='Path to the saved model')
    parser.add_argument('--dataset', nargs='?', type=str, default='quickdraw',
                        help='Dataset to use [\'pascal, camvid, ade20k, cityscapes, etc\']')
    parser.add_argument('--img_rows', nargs='?', type=int, default=128,
                        help='Height of the input image')
    parser.add_argument('--img_cols', nargs='?', type=int, default=128,
                        help='Width of the input image')

    parser.add_argument('--batch_size', nargs='?', type=int, default=1,
                        help='Batch Size')
    parser.add_argument('--split', nargs='?', type=str, default='test',
                        help='Split of dataset to test on')

    parser.add_argument('--use_cbam', dest='use_cbam', action='store_true',
                        help='Enable to use CBAM | True by default')
    parser.add_argument('--no-use_cbam', dest='use_cbam', action='store_false',
                        help='Disable to use CBAM | True by default')
    parser.set_defaults(use_cbam=True)

    parser.add_argument('--no_gt', dest='no_gt', action='store_true',
                        help='Disable verification | True by default')
    parser.add_argument('--gt', dest='no_gt', action='store_false',
                        help='Enable verification | True by default')
    parser.set_defaults(no_gt=True)

    parser.add_argument('--seed', nargs='?', type=int, default=1234,
                        help='Random seed')

    parser.add_argument('--tta', dest='tta', action='store_true',
                        help='Enable Test Time Augmentation (TTA) with horizontal flip | True by default')
    parser.add_argument('--no-tta', dest='tta', action='store_false',
                        help='Disable Test Time Augmentation (TTA) with horizontal flip | True by default')
    parser.set_defaults(tta=True)

    args = parser.parse_args()
    print(args)
    test(args)
