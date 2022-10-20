import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import json
from PIL import Image
from torchvision import transforms
from einops import rearrange
from ldm.util import instantiate_from_config
from datasets import load_dataset
import mmap
import concurrent
import pickle
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import as_completed
import requests
import hashlib
import io
import os
from simplejpeg import decode_jpeg
import simplejpeg
from PIL import Image
import time

class FolderData(Dataset):
    def __init__(self, root_dir, caption_file=None, image_transforms=[], ext="jpg") -> None:
        self.root_dir = Path(root_dir)
        self.default_caption = ""
        if caption_file is not None:
            with open(caption_file, "rt") as f:
                captions = json.load(f)
            self.captions = captions
        else:
            self.captions = None

        self.paths = list(self.root_dir.rglob(f"*.{ext}"))
        image_transforms = [instantiate_from_config(tt) for tt in image_transforms]
        image_transforms.extend([transforms.ToTensor(),
                                 transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])
        image_transforms = transforms.Compose(image_transforms)
        self.tform = image_transforms

        # assert all(['full/' + str(x.name) in self.captions for x in self.paths])

    def __len__(self):
        if self.captions is not None:
            return len(self.captions.keys())
        else:
            return len(self.paths)

    def __getitem__(self, index):
        if self.captions is not None:
            chosen = list(self.captions.keys())[index]
            caption = self.captions[chosen]
            if caption is None:
                caption = self.default_caption
            im = Image.open(self.root_dir/chosen)
        else:
            im = Image.open(self.paths[index])

        im = self.process_im(im)
        data = {"image": im}
        if self.captions is not None:
            data["txt"] = caption
        return data

    def process_im(self, im):
        im = im.convert("RGB")
        return self.tform(im)

class FolderDataLocal(Dataset):
    def __init__(self, root_dir, image_transforms=[]) -> None:
        # self.root_dir = Path(root_dir)
        self.root_dir = root_dir
        self.default_caption = ""

        ext = ['png', 'jpg', 'jpeg', 'bmp']
        self.paths = []
        # img_path = Path(f'{self.root_dir}/img/')
        # [self.paths.extend(list(img_path.rglob(f"*.{e}"))) for e in ext]
        [self.paths.extend(glob.glob(f'{self.root_dir}/img/' + '*.' + e)) for e in ext]
        self.captions = []
        for i in self.paths:
            hash = i[len(f'{self.root_dir}/img/'):].split('.')[0]
            self.captions.append(f'{self.root_dir}/txt/{hash}.txt')

        image_transforms = [instantiate_from_config(tt) for tt in image_transforms]
        image_transforms.extend([transforms.ToTensor(),
                                 transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])
        image_transforms = transforms.Compose(image_transforms)
        self.tform = image_transforms

        # assert all(['full/' + str(x.name) in self.captions for x in self.paths])

    def __len__(self):
        if self.captions is not None:
            return len(self.captions)
        else:
            return len(self.paths)

    def __getitem__(self, index):
        chosen = self.captions[index]
        with open(chosen, 'rt') as f:
            caption = f.read()
        caption = caption.strip('\n')
        im = Image.open(self.paths[index])

        im = self.process_im(im)
        data = {"image": im, "txt": caption}
        return data

    def process_im(self, im):
        im = im.convert("RGB")
        return self.tform(im)

def hf_dataset(
    name,
    image_transforms=[],
    image_column="image",
    text_column="text",
    split='train',
    image_key='image',
    caption_key='txt',
    ):
    """Make huggingface dataset with appropriate list of transforms applied
    """
    ds = load_dataset(name, split=split)
    image_transforms = [instantiate_from_config(tt) for tt in image_transforms]
    image_transforms.extend([transforms.ToTensor(),
                                transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])
    tform = transforms.Compose(image_transforms)

    assert image_column in ds.column_names, f"Didn't find column {image_column} in {ds.column_names}"
    assert text_column in ds.column_names, f"Didn't find column {text_column} in {ds.column_names}"

    def pre_process(examples):
        processed = {}
        processed[image_key] = [tform(im) for im in examples[image_column]]
        processed[caption_key] = examples[text_column]
        return processed

    ds.set_transform(pre_process)
    return ds

class TextOnly(Dataset):
    def __init__(self, captions, output_size, image_key="image", caption_key="txt", n_gpus=1):
        """Returns only captions with dummy images"""
        self.output_size = output_size
        self.image_key = image_key
        self.caption_key = caption_key
        if isinstance(captions, Path):
            self.captions = self._load_caption_file(captions)
        else:
            self.captions = captions

        if n_gpus > 1:
            # hack to make sure that all the captions appear on each gpu
            repeated = [n_gpus*[x] for x in self.captions]
            self.captions = []
            [self.captions.extend(x) for x in repeated]

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index):
        dummy_im = torch.zeros(3, self.output_size, self.output_size)
        dummy_im = rearrange(dummy_im * 2. - 1., 'c h w -> h w c')
        return {self.image_key: dummy_im, self.caption_key: self.captions[index]}

    def _load_caption_file(self, filename):
        with open(filename, 'rt') as f:
            captions = f.readlines()
        return [x.strip('\n') for x in captions]
