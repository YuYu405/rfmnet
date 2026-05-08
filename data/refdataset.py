import json
import os
import random

import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from data.utils import (
    binary_loader,
    collect_r2c_data,
    colorEnhance,
    cv_random_flip,
    randomCrop,
    randomPeper,
    randomRotation,
    rgb_loader,
)


class R2CObjData(Dataset):
    def __init__(self, data_root, mode='train', shot=1, image_size=352):
        print('this is refdataset')

        assert mode in ['train', 'val', 'test']
        self.mode = mode
        self.data_root = data_root
        self.shot = shot

        self.data_list, self.ref_images_dict_list = collect_r2c_data(
            data_root=self.data_root,
            mode=self.mode,
        )

        if self.mode == 'val' and self.shot not in [-1, 0, 5]:
            self.record_class_files()

        self.img_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.gt_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        image_path, label_path = self.data_list[index]

        image = rgb_loader(image_path)
        label = binary_loader(label_path)

        if self.mode == 'train':
            image, label = self.aug_data(image=image, label=label)

        image = self.img_transform(image)
        if self.mode == 'train':
            label = self.gt_transform(label)
        else:
            label = np.asarray(label, np.float32)

        data_dict = {
            'camo_image': image,
            'camo_gt': label,
            'camo_path_list': image_path,
        }

        if self.shot > 0 or self.shot == -1:
            class_chosen = image_path.split('/')[-1].split('-')[-2]
            file_image_class_chosen = self.ref_images_dict_list[class_chosen]
            num_aux = len(file_image_class_chosen)
            ref_idx_list = random.sample(range(num_aux), self.shot) if self.shot > 0 else list(range(num_aux))
            ref_image_list = [
                self.img_transform(rgb_loader(file_image_class_chosen[idx]))
                for idx in ref_idx_list
            ]
            data_dict['ref_sod_image_list'] = ref_image_list

        return data_dict

    def record_class_files(self):
        file_path = './data/dataset_{}shot_val.json'.format(self.shot)

        if os.path.exists(file_path):
            print('load from {}...'.format(file_path))
            with open(file_path, 'r') as f:
                self.ref_images_dict_list = json.load(f)
        else:
            print('generating {}...'.format(file_path))
            for cate in self.ref_images_dict_list.keys():
                cate_file_pairs = self.ref_images_dict_list[cate]
                assert len(cate_file_pairs) > self.shot
                rand_idxs = random.sample(range(len(cate_file_pairs)), self.shot)
                self.ref_images_dict_list[cate] = [cate_file_pairs[idx] for idx in rand_idxs]
            with open(file_path, 'w') as f:
                json.dump(self.ref_images_dict_list, f, indent=4)

    def aug_data(self, image, label):
        image, label = cv_random_flip(image, label)
        image, label = randomCrop(image, label)
        image, label = randomRotation(image, label)
        image = colorEnhance(image)
        label = randomPeper(label)
        return image, label
