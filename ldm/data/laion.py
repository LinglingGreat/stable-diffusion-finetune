import webdataset as wds
from PIL import Image
import io
import os
import torchvision
from PIL import Image
import glob
import random
import numpy as np
import pytorch_lightning as pl
from tqdm import tqdm
from omegaconf import OmegaConf
from einops import rearrange
import torch
from webdataset.handlers import warn_and_continue


from ldm.util import instantiate_from_config


def dict_collation_fn(samples, combine_tensors=True, combine_scalars=True):
    """Take a list  of samples (as dictionary) and create a batch, preserving the keys.
    If `tensors` is True, `ndarray` objects are combined into
    tensor batches.
    :param dict samples: list of samples
    :param bool tensors: whether to turn lists of ndarrays into a single ndarray
    :returns: single sample consisting of a batch
    :rtype: dict
    """
    batched = {key: [] for key in samples[0]}

    for s in samples:
        [batched[key].append(s[key]) for key in batched]

    result = {}
    for key in batched:
        if isinstance(batched[key][0], (int, float)):
            if combine_scalars:
                result[key] = np.array(list(batched[key]))
        elif isinstance(batched[key][0], torch.Tensor):
            if combine_tensors:
                result[key] = torch.stack(list(batched[key]))
        elif isinstance(batched[key][0], np.ndarray):
            if combine_tensors:
                result[key] = np.array(list(batched[key]))
        else:
            result[key] = list(batched[key])
    return result


class WebDataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, tar_base, batch_size, train=None, validation=None,
                 test=None, num_workers=4, multinode=True,
                 **kwargs):
        super().__init__(self)
        print(f'Setting tar base to {tar_base}')
        self.tar_base = tar_base
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train = train
        self.validation = validation
        self.test = test
        self.multinode = multinode

    def make_loader(self, dataset_config, train=True):
        if 'image_transforms' in dataset_config:
            image_transforms = [instantiate_from_config(tt) for tt in dataset_config.image_transforms]
        else:
            image_transforms = []

        image_transforms.extend([torchvision.transforms.ToTensor(),
                                 torchvision.transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])
        image_transforms = torchvision.transforms.Compose(image_transforms)

        if 'transforms' in dataset_config:
            transforms_config = OmegaConf.to_container(dataset_config.transforms)
        else:
            transforms_config = dict()

        transform_dict = {dkey: load_partial_from_config(transforms_config[dkey])
                if transforms_config[dkey] != 'identity' else identity
                for dkey in transforms_config}
        img_key = dataset_config.get('image_key', 'jpeg')
        transform_dict.update({img_key: image_transforms})

        shuffle = dataset_config.get('shuffle', 0)
        shardshuffle = shuffle > 0

        nodesplitter = wds.shardlists.split_by_node if self.multinode else wds.shardlists.single_node_only

        tars = os.path.join(self.tar_base, dataset_config.shards)
        dset = wds.WebDataset(
                tars,
                nodesplitter=nodesplitter,
                shardshuffle=shardshuffle).shuffle(shuffle)
        print(f'Loading webdataset with {len(dset.pipeline[0].urls)} shards.')

        dset = (dset
                .decode('pil', handler=warn_and_continue)
                .map_dict(**transform_dict)
                .batched(self.batch_size, partial=False,
                         collation_fn=dict_collation_fn)
                )

        loader = wds.WebLoader(dset, batch_size=None, shuffle=False,
                               num_workers=self.num_workers)

        return loader

    def train_dataloader(self):
        return self.make_loader(self.train)

    def val_dataloader(self):
        return self.make_loader(self.validation, train=False)

    def test_dataloader(self):
        return self.make_loader(self.test, train=False)


def example00():
    url = "pipe:aws s3 cp s3://s-datasets/laion5b/laion2B-data/000000.tar -"
    dataset = wds.WebDataset(url)
    example = next(iter(dataset))
    for k in example:
        print(k, type(example[k]))

    print(example["__key__"])
    for k in ["json", "txt"]:
        print(example[k].decode())

    image = Image.open(io.BytesIO(example["jpg"]))
    outdir = "tmp"
    os.makedirs(outdir, exist_ok=True)
    image.save(os.path.join(outdir, example["__key__"] + ".png"))


    def load_example(example):
        return {
            "key": example["__key__"],
            "image": Image.open(io.BytesIO(example["jpg"])),
            "text": example["txt"].decode(),
        }


    for i, example in tqdm(enumerate(dataset)):
        ex = load_example(example)
        print(ex["image"].size, ex["text"])
        if i >= 100:
            break


def example01():
    # the first laion shards contain ~10k examples each
    url = "pipe:aws s3 cp s3://s-datasets/laion5b/laion2B-data/{000000..000002}.tar -"

    batch_size = 3
    shuffle_buffer = 10000
    dset = wds.WebDataset(
            url,
            nodesplitter=wds.shardlists.split_by_node,
            shardshuffle=True,
            )
    dset = (dset
            .shuffle(shuffle_buffer, initial=shuffle_buffer)
            .decode('pil', handler=warn_and_continue)
            .batched(batch_size, partial=False,
                collation_fn=dict_collation_fn)
            )

    num_workers = 2
    loader = wds.WebLoader(dset, batch_size=None, shuffle=False, num_workers=num_workers)

    batch_sizes = list()
    keys_per_epoch = list()
    for epoch in range(5):
        keys = list()
        for batch in tqdm(loader):
            batch_sizes.append(len(batch["__key__"]))
            keys.append(batch["__key__"])

        for bs in batch_sizes:
            assert bs==batch_size
        print(f"{len(batch_sizes)} batches of size {batch_size}.")
        batch_sizes = list()

        keys_per_epoch.append(keys)
        for i_batch in [0, 1, -1]:
            print(f"Batch {i_batch} of epoch {epoch}:")
            print(keys[i_batch])
        print("next epoch.")


if __name__ == "__main__":
    example01()