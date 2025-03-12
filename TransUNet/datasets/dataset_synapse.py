import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from pipeline.video_input import frame_split


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()

    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape

        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)

        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None, val_ratio=0.2, seed=42, video_path=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        #self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir
        self.video_path = video_path

        if self.video_path and not os.listdir(self.data_dir):
            print(f'No .npz files found in {self.data_dir}. Processing video...')
            frame_split(self.video_path, self.data_dir)

        all_samples = open(os.path.join(list_dir, "train.txt")).readlines()
        all_samples = [s.strip() for s in all_samples]

        random.seed(seed)
        random.shuffle(all_samples)

        train_split = int(len(all_samples) * (1 - val_ratio))

        if split == "train":
            self.sample_list = all_samples[:train_split]  
        elif split == "val":
            self.sample_list = all_samples[train_split:]  
        elif split == "test_vol":
            self.sample_list = open(os.path.join(list_dir, "test_vol.txt")).readlines()
            self.sample_list = [s.strip() for s in self.sample_list]  
        else:
            raise ValueError("Split debe ser 'train', 'val' o 'test'.")


    def __len__(self):
        return len(self.sample_list)
    
    def __getitem__(self, idx):
        slice_name = self.sample_list[idx].strip('\n')
        data_path = os.path.join(self.data_dir, slice_name+'.npz')
        data = np.load(data_path)
        image, label = data['image'], data['label']
        
        if self.split != 'test_vol':
            coords = data['insertion_coords']

        if image.ndim == 3:
            image = np.mean(image, axis=2)

        print(image.shape)

        x, y = image.shape
        if x != 224 or y != 224:
            image = zoom(image, (224 / x, 224 / y), order=3)  
            label = zoom(label, (224 / x, 224 / y), order=0)

        image = np.expand_dims(image, axis=0)
        image = np.repeat(image, 3, axis=0)    

        sample = {'image': image, 'label': label} 
        if self.split != 'test_vol':
            sample['coords'] = coords

        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')

        return sample
