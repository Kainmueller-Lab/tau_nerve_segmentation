import torch
import numpy as np
from torchvision.transforms import functional as func_transforms
from imgaug import augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import glob
import os
import zarr


class ToTensor(object):
    def __call__(self, image, target):
        image = func_transforms.to_tensor(image)
        target = torch.as_tensor(np.asarray(target), dtype=torch.long)
        return image, target


class CustomDataset(Dataset):
    def __init__(self, data_root, mode="train", padding_size=None,
                 net_input_size=None):
        self.samples = glob.glob(data_root)
        self.label_dict = self.get_data_dict("label")
        self.raw_dict = self.get_data_dict("raw")
        self.mode = mode
        self.padding_size = padding_size
        self.net_input_size = net_input_size
        self.define_augmentation()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        raw, labels = self.load_sample(self.samples[idx])

        if self.mode == "train":
            raw, labels = self.pad_sample(raw, labels)
            raw, labels = self.augment_sample(raw, labels)
            raw, labels = self.crop_sample(raw, labels)
        else:
            raw, _ = self.pad_sample(raw, None)

        raw, labels = self.to_tensor(raw, labels)

        return raw, labels

    def define_augmentation(self):
        self.transform = iaa.Identity
        self.crop = None
        self.pad = None

        self.transform = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.Affine(rotate=(-45, 45)),  # rotate by -45 to 45 degrees (affects segmaps)
        ], random_order=True)

        if self.net_input_size is not None:
            self.crop = iaa.CropToFixedSize(width=self.net_input_size[1], height=self.net_input_size[0])
        else:
            self.crop = None

        if self.padding_size is not None:
            self.pad = iaa.Pad(px=(self.padding_size, self.padding_size, self.padding_size, self.padding_size),
                               keep_size=False)
        else:
            self.crop = None

        self.to_tensor = ToTensor()

    def load_sample(self, filename):
        raw = self.raw_dict[filename]
        labels = np.array(self.label_dict[filename]).astype(np.int16)
        return raw, labels

    def get_data_dict(self, data_type):
        data_dict = {}
        print(self.samples)
        for sample in data_type + "/" + self.samples:
            data_dict[sample] = np.load(sample)
        return data_dict

    def pad_sample(self, raw, labels):
        if self.pad is not None:
            if labels is None:
                raw = self.pad(image=raw)
            else:
                labels = SegmentationMapsOnImage(labels, shape=raw.shape)
                raw, labels = self.pad(image=raw, segmentation_maps=labels)
                labels = labels.get_arr()
        return raw, labels

    def crop_sample(self, raw, labels):
        if self.crop is not None:
            labels = SegmentationMapsOnImage(labels, shape=raw.shape)
            raw, labels = self.crop(image=raw, segmentation_maps=labels)
            labels = labels.get_arr()
        return raw, labels

    def augment_sample(self, raw, labels):
        # this code makes sure that the same geometric augmentations are applied
        # to both the raw image and the label image
        labels = SegmentationMapsOnImage(labels, shape=raw.shape)
        raw, labels = self.transform(image=raw, segmentation_maps=labels)
        labels = labels.get_arr()

        # some pytorch version have problems with negative indices introduced by e.g. flips
        # just copying fixes this
        labels = labels.copy()
        raw = raw.copy()

        return raw, labels


def main():
    data_root = "data/cell/"
    dataset = CustomDataset(data_root, "test")
    test_dataloader = DataLoader(dataset)
    image = iter(test_dataloader)
    print(image)


main()
