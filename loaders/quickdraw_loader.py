import os
import random
import gc
import ast
import cv2
import torch
import pickle
import pandas as pd
import numpy as np

from torch.utils import data

from misc.utils import recursive_glob


class quickdrawLoader(data.Dataset):
    def __init__(self, root, split="train", version="simplified", is_transform=True,
                 img_size=(256, 256), augmentations=None,
                 no_gt=False, train_fold_num=0, num_train_folds=10, num_val=300, seed=1234):

        self.root = root
        self.split = split
        self.version = version
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.no_gt = no_gt
        self.n_classes = 340
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.mean_rgb = [0.485, 0.456, 0.406]
        self.std_rgb = [0.229, 0.224, 0.225] # torchvision pretrained image transform
        self.files = {}

        if self.split == 'test':
            csv_path = os.path.join(self.root, '{}_{}.csv'.format(self.split, self.version))
            test_df = pd.read_csv(csv_path, usecols=['key_id', 'drawing'])[['key_id', 'drawing']]
            test_dict = test_df.to_dict('index')
            gc.collect()
            self.files[split] = test_dict
            fs = recursive_glob(rootdir=os.path.join(self.root, '{}_{}'.format('train', self.version)), suffix='.csv')
        else:
            torch.manual_seed(seed)
            csv_path = os.path.join(self.root, '{}_{}'.format(self.split.replace('val', 'train'), self.version))
            suffix = '_{}_{}_{}.pkl'.format(self.split, train_fold_num, num_train_folds) if self.split == 'train' else '_{}_{}.pkl'.format(self.split, num_val)
            fs = recursive_glob(rootdir=csv_path, suffix=suffix)
            if len(fs) == 0:
                suffix = '.csv'
                fs = recursive_glob(rootdir=csv_path, suffix=suffix)
            train_dicts = {}
            accum = 0
            for i, f in enumerate(fs):
                if suffix == '.csv':
                    train_df = pd.read_csv(f, usecols=['key_id', 'drawing', 'recognized', 'word'])[['key_id', 'drawing', 'recognized', 'word']]
                    train_dict = train_df.to_dict('index')
                    num_train = (len(train_dict) - num_val) // num_train_folds
                    print('({:3d}) {:25s}: {:7d}'.format(i, os.path.basename(f).split('.')[0].split('_')[0].replace(' ', '_'), len(train_dict)))
                else:
                    with open(f, 'rb') as pkl_f:
                        train_dict = pickle.load(pkl_f)
                if num_train_folds > 0 and suffix == '.csv':
                    #"""
                    class_name = os.path.basename(f).split('.')[0]
                    rp = torch.randperm(len(train_dict)).tolist()
                    total_num_train = len(train_dict) - num_val
                    for fold_num in range(num_train_folds):
                        start_index = fold_num * num_train
                        end_index = (fold_num+1) * num_train if fold_num < num_train_folds-1 else total_num_train
                        selected_train_indices = rp[start_index:end_index]
                        part_train_dict = {j: train_dict[key_index] for j, key_index in enumerate(selected_train_indices)}
                        part_train_pkl_filename = os.path.join(csv_path, '{}_{}_{}_{}.pkl'.format(class_name, 'train', fold_num, num_train_folds))
                        with open(part_train_pkl_filename, 'wb') as pkl_f:
                            pickle.dump(part_train_dict, pkl_f, protocol=pickle.HIGHEST_PROTOCOL)

                    selected_val_indices = rp[-num_val:]
                    part_val_dict = {j: train_dict[key_index] for j, key_index in enumerate(selected_val_indices)}
                    part_val_pkl_filename = os.path.join(csv_path, '{}_{}_{}.pkl'.format(class_name, 'val', num_val))
                    with open(part_val_pkl_filename, 'wb') as pkl_f:
                        pickle.dump(part_val_dict, pkl_f, protocol=pickle.HIGHEST_PROTOCOL)
                    #"""
                    train_dict_len = num_train
                else:
                    train_dict_len = len(train_dict)
                train_dict = {accum+j: train_dict[j] for j in range(train_dict_len)}
                train_dicts.update(train_dict)
                accum = accum + train_dict_len
                gc.collect()
            self.files[split] = train_dicts

        self.class_num2name = [os.path.basename(f).split('.')[0].split('_')[0].replace(' ', '_') for f in fs]
        self.ignore_index = -1

        self.class_name2num = dict(zip(self.class_num2name, range(self.n_classes)))

        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, csv_path))

        print("Found %d %s images" % (len(self.files[split]), split))

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        info = self.files[self.split][index]

        img = self.decode_drawing(ast.literal_eval(info['drawing']))
        key_id = [info['key_id']]
        recognized = torch.tensor([1.]).float()

        if not self.no_gt:
            if self.augmentations is not None:
                img = np.array(self.augmentations(img), dtype=np.uint8)
            lbl = self.class_name2num[info['word'].replace(' ', '_')]
        else:
            lbl = self.ignore_index
        lbl = np.array([lbl], dtype=np.int32)

        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        return img, lbl, recognized, key_id

    def transform(self, img, lbl):
        if img.shape[0] != self.img_size[0] or img.shape[1] != self.img_size[1]:
            img = cv2.resize(img, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_LINEAR) # cv2.resize shape: (W, H)

        if len(img.shape) == 3:
            img = img[:, :, ::-1] # RGB -> BGR

        img = img.astype(np.float64) / 255.0 # Rescale images from [0, 255] to [0, 1]
        img = (img - self.mean_rgb) / self.std_rgb

        if len(img.shape) == 3:
            img = img.transpose(2, 0, 1) # NHWC -> NCHW
        else:
            img = np.expand_dims(img, axis=0)

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl

    def decode_drawing(self, raw_strokes, line_thickness=5, time_color=True, part_color=True, num_channels=3):
        img = np.zeros((256, 256, num_channels), dtype=np.uint8)
        for t, stroke in enumerate(raw_strokes):
            part_num = int(float(t) / len(raw_strokes) * num_channels)
            for i in range(len(stroke[0]) - 1):
                color = 255 - min(t, 20) * 10 if time_color else 255
                if part_color:
                    if part_num == 1:
                        color = (0, color, color)
                    elif part_num == 2:
                        color = (0, 0, color)
                    else:#if part_num == 0:
                        color = (color, color, color)
                cv2.line(img, (stroke[0][i], stroke[1][i]), (stroke[0][i+1], stroke[1][i+1]), color, line_thickness, cv2.LINE_AA)
        return img

    def encode_pred_name(self, preds):
        name_str = self.class_num2name[preds[0]]
        for i in range(1, preds.size(0)):
            name_str = name_str + ' ' + self.class_num2name[preds[i]]
        return name_str



if __name__ == '__main__':
    local_path = '../'
    dst = quickdrawLoader(local_path, split="train", train_fold_num=0, num_train_folds=10, num_val=300)
    loader = data.DataLoader(dst, batch_size=512)
