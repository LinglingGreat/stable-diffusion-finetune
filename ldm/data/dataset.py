import numpy as np
import torch
import mmap
import concurrent
from torch.utils.data import Dataset
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

# Does this work with other block_sizes? doesn't seem to.
class FbDataset(Dataset):
    def __init__(self, block_size, map_file, max_samples=None, skip=0):
        self.npz = np.memmap(map_file, mode="r", dtype="uint16").reshape((-1, block_size))
        self.samples = self.npz.shape[0]
        if max_samples is not None:
            self.samples = min(self.samples, int(max_samples))
        self.skip = skip

    def __len__(self):
        return self.samples

    def __getitem__(self, _id):
        nth = _id + self.skip
        data = torch.tensor(self.npz[nth].astype(np.int64))
        return (data[:-1], data[1:])

class ShardedDataset(Dataset):
    def __init__(self, block_size, map_file, world_size=1, rank=0, skip=0):
        self.npz = np.memmap(map_file, mode="r", dtype="uint16").reshape((-1, block_size))
        #might want to pad later
        self.npz = self.npz[:self.npz.shape[0] - (self.npz.shape[0] % world_size)]
        #shard
        self.npz = self.npz[rank::world_size]
        self.samples = self.npz.shape[0]
        self.skip = skip

    def __len__(self):
        return self.samples

    def __getitem__(self, _id):
        nth = _id + self.skip
        data = torch.tensor(self.npz[nth].astype(np.int64))
        return (data[:-1], data[1:])

def get_prng(seed):
    prng = np.random.RandomState(seed)
    prng.seed(seed)
    return prng

class BucketManager:
    def __init__(self, bucket_file, valid_ids=None, max_size=(640,512), divisible=64, step_size=8, min_dim=256, base_res=(512,512), bsz=1, world_size=1, global_rank=0, max_ar_error=4, seed=69, dim_limit=1024, res_dropout=0.0, debug=False):
        with open(bucket_file, "rb") as fh:
            self.res_map = pickle.load(fh)
        if valid_ids is not None:
            new_res_map = {}
            valid_ids = set(valid_ids)
            for k, v in self.res_map.items():
                if k in valid_ids:
                    new_res_map[k] = v
            self.res_map = new_res_map
        self.max_size = max_size
        self.f = 8
        self.max_tokens = (max_size[0]/self.f) * (max_size[1]/self.f)
        self.div = divisible
        self.min_dim = min_dim
        self.dim_limit = dim_limit
        self.base_res = base_res
        self.res_dropout = res_dropout
        self.bsz = bsz
        self.world_size = world_size
        self.global_rank = global_rank
        self.max_ar_error = max_ar_error
        self.prng = get_prng(seed)
        epoch_seed = self.prng.tomaxint() % (2**32-1)
        self.epoch_prng = get_prng(epoch_seed) # separate prng for sharding use for increased thread resilience
        self.epoch = None
        self.left_over = None
        self.batch_total = None
        self.batch_delivered = None

        self.debug = debug

        self.gen_buckets()
        self.assign_buckets()
        self.start_epoch()

    def gen_buckets(self):
        if self.debug:
            timer = time.perf_counter()
        resolutions = []
        aspects = []
        w = self.min_dim
        # w * self.min_dim < max_size[0] * max_size[1] and w <= self.dim_limit
        while (w/self.f) * (self.min_dim/self.f) <= self.max_tokens and w <= self.dim_limit:
            h = self.min_dim
            got_base = False
            # w * (h+self.div) < max_size[0] * max_size[1] and h+self.div <= self.dim_limit
            while (w/self.f) * ((h+self.div)/self.f) <= self.max_tokens and (h+self.div) <= self.dim_limit:
                if w == self.base_res[0] and h == self.base_res[1]:
                    got_base = True
                h += self.div
            if (w != self.base_res[0] or h != self.base_res[1]) and got_base:
                resolutions.append(self.base_res)
                aspects.append(1)
            resolutions.append((w, h))
            aspects.append(float(w)/float(h))
            w += self.div
        h = self.min_dim
        while (h/self.f) * (self.min_dim/self.f) <= self.max_tokens and h <= self.dim_limit:
            w = self.min_dim
            got_base = False
            while (h/self.f) * ((w+self.div)/self.f) <= self.max_tokens and (w+self.div) <= self.dim_limit:
                if w == self.base_res[0] and h == self.base_res[1]:
                    got_base = True
                w += self.div
            resolutions.append((w, h))
            aspects.append(float(w)/float(h))
            h += self.div
        res_map = {}
        for i, res in enumerate(resolutions):
            res_map[res] = aspects[i]
        self.resolutions = sorted(res_map.keys(), key=lambda x: x[0] * 4096 - x[1])
        self.aspects = np.array(list(map(lambda x: res_map[x], self.resolutions)))
        self.resolutions = np.array(self.resolutions)
        if self.debug:
            timer = time.perf_counter() - timer
            print(f"resolutions:\n{self.resolutions}")
            print(f"aspects:\n{self.aspects}")
            print(f"gen_buckets: {timer:.5f}s")

    def assign_buckets(self):
        if self.debug:
            timer = time.perf_counter()
        self.buckets = {}
        self.aspect_errors = []
        skipped = 0
        skip_list = []
        for post_id in self.res_map.keys():
            w, h = self.res_map[post_id]
            aspect = float(w)/float(h)
            bucket_id = np.abs(self.aspects - aspect).argmin()
            if bucket_id not in self.buckets:
                self.buckets[bucket_id] = []
            error = abs(self.aspects[bucket_id] - aspect)
            if error < self.max_ar_error:
                self.buckets[bucket_id].append(post_id)
                if self.debug:
                    self.aspect_errors.append(error)
            else:
                skipped += 1
                skip_list.append(post_id)
        for post_id in skip_list:
            del self.res_map[post_id]
        if self.debug:
            timer = time.perf_counter() - timer
            self.aspect_errors = np.array(self.aspect_errors)
            print(f"skipped images: {skipped}")
            print(f"aspect error: mean {self.aspect_errors.mean()}, median {np.median(self.aspect_errors)}, max {self.aspect_errors.max()}")
            for bucket_id in reversed(sorted(self.buckets.keys(), key=lambda b: len(self.buckets[b]))):
                print(f"bucket {bucket_id}: {self.resolutions[bucket_id]}, aspect {self.aspects[bucket_id]:.5f}, entries {len(self.buckets[bucket_id])}")
            print(f"assign_buckets: {timer:.5f}s")

    def start_epoch(self, world_size=None, global_rank=None):
        if self.debug:
            timer = time.perf_counter()
        if world_size is not None:
            self.world_size = world_size
        if global_rank is not None:
            self.global_rank = global_rank

        # select ids for this epoch/rank
        index = np.array(sorted(list(self.res_map.keys())))
        index_len = index.shape[0]
        index = self.epoch_prng.permutation(index)
        index = index[:index_len - (index_len % (self.bsz * self.world_size))]
        #print("perm", self.global_rank, index[0:16])
        index = index[self.global_rank::self.world_size]
        self.batch_total = index.shape[0] // self.bsz
        assert(index.shape[0] % self.bsz == 0)
        index = set(index)

        self.epoch = {}
        self.left_over = []
        self.batch_delivered = 0
        for bucket_id in sorted(self.buckets.keys()):
            if len(self.buckets[bucket_id]) > 0:
                self.epoch[bucket_id] = np.array([post_id for post_id in self.buckets[bucket_id] if post_id in index], dtype=np.int64)
                self.prng.shuffle(self.epoch[bucket_id])
                self.epoch[bucket_id] = list(self.epoch[bucket_id])
                overhang = len(self.epoch[bucket_id]) % self.bsz
                if overhang != 0:
                    self.left_over.extend(self.epoch[bucket_id][:overhang])
                    self.epoch[bucket_id] = self.epoch[bucket_id][overhang:]
                if len(self.epoch[bucket_id]) == 0:
                    del self.epoch[bucket_id]

        bucket_ids = list(self.epoch.keys())
        resolution = dict()
        for k in bucket_ids:
            resolution[k] = self.resolutions[k]
        bucket_num = {k:len(v) for k, v in self.epoch.items()}
        print(f"All bucket resolution: {resolution}")
        print(f"All bucket num: {bucket_num}")
        if self.debug:
            timer = time.perf_counter() - timer
            count = 0
            for bucket_id in self.epoch.keys():
                count += len(self.epoch[bucket_id])
            print(f"correct item count: {count == len(index)} ({count} of {len(index)})")
            print(f"start_epoch: {timer:.5f}s")

    def get_batch(self):
        if self.debug:
            timer = time.perf_counter()
        # check if no data left or no epoch initialized
        if self.epoch is None or self.left_over is None or (len(self.left_over) == 0 and not bool(self.epoch)) or self.batch_total == self.batch_delivered:
            self.start_epoch()

        found_batch = False
        batch_data = None
        resolution = self.base_res
        while not found_batch:
            bucket_ids = list(self.epoch.keys())
            if len(self.left_over) >= self.bsz:
                bucket_probs = [len(self.left_over)] + [len(self.epoch[bucket_id]) for bucket_id in bucket_ids]
                bucket_ids = [-1] + bucket_ids
            else:
                bucket_probs = [len(self.epoch[bucket_id]) for bucket_id in bucket_ids]
            bucket_probs = np.array(bucket_probs, dtype=np.float32)
            bucket_lens = bucket_probs
            bucket_probs = bucket_probs / bucket_probs.sum()
            bucket_ids = np.array(bucket_ids, dtype=np.int64)
            if bool(self.epoch):
                chosen_id = int(self.prng.choice(bucket_ids, 1, p=bucket_probs)[0])
            else:
                chosen_id = -1

            if chosen_id == -1:
                # using leftover images that couldn't make it into a bucketed batch and returning them for use with basic square image
                self.prng.shuffle(self.left_over)
                batch_data = self.left_over[:self.bsz]
                self.left_over = self.left_over[self.bsz:]
                found_batch = True
            else:
                if len(self.epoch[chosen_id]) >= self.bsz:
                    # return bucket batch and resolution
                    batch_data = self.epoch[chosen_id][:self.bsz]
                    self.epoch[chosen_id] = self.epoch[chosen_id][self.bsz:]
                    resolution = tuple(self.resolutions[chosen_id])
                    found_batch = True
                    if len(self.epoch[chosen_id]) == 0:
                        del self.epoch[chosen_id]
                else:
                    # can't make a batch from this, not enough images. move them to leftovers and try again
                    self.left_over.extend(self.epoch[chosen_id])
                    del self.epoch[chosen_id]

            assert(found_batch or len(self.left_over) >= self.bsz or bool(self.epoch))

        if self.debug:
            timer = time.perf_counter() - timer
            print(f"bucket probs: " + ", ".join(map(lambda x: f"{x:.2f}", list(bucket_probs*100))))
            print(f"chosen id: {chosen_id}")
            print(f"batch data: {batch_data}")
            print(f"resolution: {resolution}")
            print(f"get_batch: {timer:.5f}s")

        if self.res_dropout > 0.0 and self.prng.random() < self.res_dropout:
            half_round_res = lambda x: int(max(1,((x/2.0)/64.+0.5)))*64
            resolution = tuple(map(half_round_res, resolution))
            if self.prng.random() < 0.25:
                resolution = tuple(map(half_round_res, resolution))

        self.batch_delivered += 1
        return (batch_data, resolution)

    def generator(self):
        if self.batch_delivered >= self.batch_total:
            self.start_epoch()
        while self.batch_delivered < self.batch_total:
            yield self.get_batch()

def inner_transform(data):
    data = CPUTransforms.scale(data, 512)
    data = CPUTransforms.randomcrop(data, 512)
    return data

class ShardedImageDataset(Dataset):
    def __init__(self, dataset_path: str, tags_db:str, name:str, index_path:str=None, shuffle=False, metadata_path=None, threads=None, inner_transform=inner_transform,
        outer_transform=None, skip=0, bsz=256, world_size=1, local_rank=0, global_rank=0, resolution_pkl=None, max_size=(640,512), min_dim=256, dim_limit=1024, divisible=64,
        bucket_seed=69, res_dropout=0.0, device="cpu"):
        self.skip = skip
        self.threads = threads
        self.bsz = bsz
        self.epoch_target = 1
        #for one by one transforms because images can't be batched
        self.inner_transform = inner_transform
        #for batched transforms after images become batchable
        self.outer_transform = outer_transform
        self.dataset_path = Path(dataset_path)
        if index_path is None:
            self.index_path = self.dataset_path / f"{name}.index"
        else:
            self.index_path = Path(index_path)
        with open(tags_db, "rb") as f:
            self.db_tags = pickle.load(f)
            
        self.pointer_path = self.dataset_path / f"{name}.pointer"
        self.dataset_path = self.dataset_path / f"{name}.ds"
        self.world_size = world_size
        self.local_rank = local_rank
        self.global_rank = global_rank
        self.device = device
        with open(self.index_path, 'rb') as f:
            self.index = pickle.load(f)

        self.bucket_manager = None
        if resolution_pkl is not None:
            valid_ids = list(np.array(self.index)[:, 2])
            self.bucket_manager = BucketManager(resolution_pkl, min_dim=min_dim, dim_limit=dim_limit, max_size=max_size, divisible=divisible,
            valid_ids=valid_ids, bsz=bsz, world_size=world_size, global_rank=global_rank, seed=bucket_seed, res_dropout=res_dropout)
            del valid_ids

        if metadata_path:
            with open(metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)

        with open(self.dataset_path, mode="r") as file_obj:
            self.mmap = mmap.mmap(file_obj.fileno(), length=0, access=mmap.ACCESS_READ)

        #precompute pointer lookup dict for faster random read
        if not self.pointer_path.is_file():
            self.pointer_lookup = {}
            for t in tqdm(self.index):
                offset, length, id = t
                self.pointer_lookup[id] = (offset, length)

            with open(self.pointer_path, 'wb') as f:
                pickle.dump(self.pointer_lookup, f)

        else:
            with open(self.pointer_path, 'rb') as f:
                self.pointer_lookup = pickle.load(f)
        #make so metadata is shardable by world_size(num_gpus)
        #and batch_size
        self.original_index = self.index
        #self.shard(shuffle=shuffle)
        #override possible gil locks by making the index map an nparray
        self.index = np.array(self.index)
        self.ids = self.index.transpose(1, 0)[2]
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.threads)

    def __len__(self):
        if self.bucket_manager is not None:
            return self.bucket_manager.batch_total * self.epoch_target
        return len(self.index) // self.bsz

    def shard(self, shuffle=False, epoch=1, seed=69):
        #get numpy random state
        state = np.random.get_state()
        self.epoch_target = epoch
        #set numpy seed
        np.random.seed(seed)
        #use this function to shuffle every new epoch as well.
        shuffled_indexes = []
        if shuffle:
            #shuffle on the epoch boundries
            for _ in range(epoch):
                #shuffle the index
                shuffled = np.random.permutation(self.index)
                #append the index
                shuffled_indexes.append(shuffled)
                #del shuffled
            #concatenate the indexes
            self.index = np.concatenate(shuffled_indexes)
        #del shuffled_indexes

        self.index = self.index[:len(self.index) - (len(self.index) % (self.bsz * self.world_size))]
        self.index = self.index[self.global_rank::self.world_size]
        #reset numpy random state
        np.random.set_state(state)
    
    def __getitem__(self, key):
        if self.bucket_manager is not None:
            db_ids, res = self.bucket_manager.get_batch()
            ids_res = list(map(lambda x: (x, res), db_ids))
            tensors = self.executor.map(self.read_from_id_bucket, ids_res)
        else:
            key = self.skip + key
            keys = [*range(key, key+self.bsz)]
            tensors = self.executor.map(self.read_from_index_key, keys)
        tensors = list(tensors)
        #make sure these operations are fast!
        ids = [t[1] for t in tensors]
        tensors = torch.stack([t[0] for t in tensors])
        if self.device == "cuda":
            tensors = tensors.to(self.local_rank)

        tensors = tensors.float()#permute#(0, 3, 1, 2).float() / 255.0
        tensors = tensors / 127.5 - 1
        #####################################
        if self.outer_transform:
            tensors = self.outer_transform(tensors)
            
        # return tensors, ids
        captions = []
        for id in ids:
            caption = self.db_tags[id.item()]
            captions.append(caption)
        return {"image": tensors, "txt": captions}

    def read_from_index_key(self, key):
        offset, size, id = self.index[key]
        data = self.mmap[offset:offset+size]
        data = decode_jpeg(data)
        if self.inner_transform:
            data = self.inner_transform(data)

        data = torch.from_numpy(data)#.permute(2, 0, 1)
        return data, id

    def read_from_id_bucket(self, entry):
        id, res = entry
        offset, size = self.pointer_lookup[id]
        data = self.mmap[offset:offset+size]
        data = decode_jpeg(data)
        data = CPUTransforms.scale(data, res, preserve_aspect=True)
        data = CPUTransforms.randomcrop(data, res)
        if data.shape[0] != res[1] or data.shape[1] != res[0]:
            data = CPUTransforms.scale(data, res)
        data = torch.from_numpy(data)#.permute(2, 0, 1)
        return data, id

    def read_from_id(self, id, decode=True):
        offset, size = self.pointer_lookup[id]
        data = self.mmap[offset:offset+size]
        if decode:
            data = decode_jpeg(data)

        data = torch.from_numpy(data)#.permute(2, 0, 1)
        return data

    def get_metadata(self, id):
        return self.metadata[id]

class CPUTransforms():
    def __init__(self, threads=None):
        self.threads=None

    @staticmethod
    def scale(data, res, pil=True, preserve_aspect=False):
        #scale can be an int or a tuple(h, w)
        #if it's int preserve aspect ratio
        #preserve aspect allows keeping aspect ratio with tuple, allowing one dimension to extend past the resolution boundary
        #use opencv2
        #data.shape = (h, w, c)
        h, w = data.shape[:2]
        #w, h = data.size

        if isinstance(res, int):
            if h > w:
                #get the scale needed to make the width match the target
                scale = res / w
                hw = (res, int(h*scale))

            elif h == w:
                hw = (res, res)
            
            else:
                #get the scale needed to make the height match the target
                scale = res / h
                hw = (int(w*scale), res)
        elif preserve_aspect:
            r_w, r_h = res
            r_ar = float(r_w)/float(r_h)
            ar = float(w)/float(h)
            s_w, s_h = float(r_w) / float(w), float(r_h) / float(h)
            if r_ar >= 1:
                if ar >= r_ar:
                    # image wider => conform to height and allow wider image
                    hw = (r_h, int(w * s_h))
                else:
                    # target wide, imager tall => conform to width and allow taller image
                    hw = (int(h * s_w), r_w)
            else:
                if ar < r_ar:
                    # image taller => conform to width and allow taller image
                    hw = (int(h * s_w), r_w)
                else:
                    # target tall, image wide => conform to height and allow wider image
                    hw = (r_h, int(w * s_h))
        else:
            hw = (res[1], res[0])

        if pil:
            data = Image.fromarray(data)
            data = data.resize((hw[1], hw[0]), Image.LANCZOS)
            data = np.asarray(data)
        else:
            data = cv2.resize(data, (hw[1], hw[0]), interpolation=cv2.INTER_AREA)
        return data
    
    @staticmethod
    def centercrop(data, res: int):
        h_offset = (data.shape[0] - res) // 2
        w_offset = (data.shape[1] - res) // 2
        data = data[h_offset:h_offset+res, w_offset:w_offset+res]
        return data

    @staticmethod
    def cast_to_rgb(data, pil=False):
        if len(data.shape) < 3:
            data = np.expand_dims(data, axis=2)
            data = np.repeat(data, 3, axis=2)
            return data
        if data.shape[2] == 1:
            data = np.repeat(data, 3, axis=2)
            return data
        if data.shape[2] == 3:
            return data
        if data.shape[2] == 4:
            #Alpha blending, remove alpha channel and blend in with white
            png = Image.fromarray(data) # ->Fails here because image is uint16??

            background = Image.new('RGBA', png.size, (255,255,255))
            alpha_composite = Image.alpha_composite(background, png)
            data = np.asarray(alpha_composite)
            '''
            data = data.astype(np.float32)
            data = data / 255.0
            alpha = data[:,:,[3,3,3]]
            data  = data[:,:,:3]
            ones = np.ones_like(data)
            data = (data * alpha) + (ones * (1-alpha))
            data = data * 255.0
            data = np.clip(data, 0, 255)
            data = data.astype(np.uint8)
            '''
            return data
        else:
            return data

    @staticmethod
    def randomcrop(data, res):
        if isinstance(res, tuple):
            res_w, res_h = res
        else:
            res_w, res_h = res, res
        h, w = data.shape[:2]
        if h - res_h > 0:
            h_offset = np.random.randint(0, h - res_h)
        else:
            h_offset = 0

        if w - res_w > 0:
            w_offset = np.random.randint(0, w - res_w)
        else:
            w_offset = 0
        data = data[h_offset:h_offset+res_h, w_offset:w_offset+res_w]
        return data

class ImageDatasetBuilder():
    def __init__(self, folder_path, name, dataset=True, index=True, metadata=False, threads=None, block_size=4096, align_fs_blocks=True):
        self.folder_path = Path(folder_path)
        self.dataset_name = name + ".ds"
        self.index_name = name + ".index"
        self.metadata_name = name + ".metadata"
        self.index_name_temp = name + ".temp.index"
        self.metadata_name_temp = name + ".temp.metadata"
        self.dataset_path = self.folder_path / self.dataset_name
        self.index_path = self.folder_path / self.index_name
        self.metadata_path = self.folder_path / self.metadata_name
        self.index_path_temp = self.folder_path / self.index_name_temp
        self.metadata_path_temp = self.folder_path / self.metadata_name_temp
        self.open_dataset = dataset
        self.open_index = index
        self.open_metadata = metadata
        self.dataset = None
        self.index = None
        self.metadata = None
        self.threads = threads
        self.block_size = block_size
        self.align_fs_blocks = align_fs_blocks

    @property
    def is_open(self):
        self.dataset is not None or self.index is not None or self.metadata is not None

    @property
    def is_close(self):
        self.dataset is None or self.index is None

    @property
    def biggest_id(self):
        try:
            return np.max(self.np_index[:, 2])
        except:
            return -1
            
    @property
    def biggest_item(self):
        try:
            return np.max(self.np_index[:, 1])
        except:
            return -1
    @property
    def total_ids(self):
        try:
            return len(self.np_index)
        except:
            return -1

    @property
    def np_index(self):
        return np.array(self.index)

    def build(self):
        #be careful with not nuking the files if they exist
        if self.is_open:
            raise Exception("Dataset already built")
        
        self.folder_path.mkdir(parents=True, exist_ok=True)
        if self.open_dataset:
            self.dataset = open(self.dataset_path, mode="ab+")
            self.dataset.flush()

        if self.open_index:
            self.index = []

        if self.open_metadata:
            self.metadata = {}

    def open(self, overwrite=False):
        if overwrite is False and self.is_open:
            raise Exception("A dataset is already open! If you wish to continue set overwrite to True.")

        if overwrite is True:
            self.close(silent=True)
            self.flush_index(silent=True)
            self.flush_metadata(silent=True)
            print("Dataset closed and flushed.")
        
        if self.open_dataset and self.dataset_path.is_file():
            self.dataset = open(self.dataset_path, mode="ab+")
        else:
            raise Exception("Dataset file not found at {}".format(self.dataset_path))
            
        if self.open_index and self.index_path.is_file():
            with open(self.index_path, 'rb') as f:
                self.index = pickle.load(f)
        else:
            raise Exception("Index file not found at {}".format(self.index_path))
        
        if self.open_metadata and self.metadata_path.is_file():
            with open(self.metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
        else:
            raise Exception("Metadata file not found at {}".format(self.metadata_path))

    def operate(self, operation, batch, identities, metadata=None, executor=concurrent.futures.ThreadPoolExecutor, use_tqdm=False, **kwargs):
        executor = executor(max_workers=self.threads)
        futures = executor.map(operation, batch)
        if use_tqdm:
            futures = tqdm(futures, total=len(batch), leave=False)
        futures = list(futures)

        for data, identity in zip(futures, identities):
            self.write(data, identity)
    
    def encode_op(self, data):
        if simplejpeg.is_jpeg(data):
            try:
                simplejpeg.decode(data)
            except:
                return None
        else:
            data = Image.open(io.BytesIO(data))
            data = np.asarray(data)
            data = simplejpeg.encode_jpeg(data, quality=91)

        return data

    def url_op(self, url, md5):
        result = requests.get(url)
        for _ in range(5):
            if result.status_code == 200:
                break
        
        if result.status_code != 200:
            return None

        data = result.content
        saved_md5 = hashlib.md5(data)
        if saved_md5 != md5:
            return None
        data = self.encode_op(data)
        return data

    def write(self, data, identity, metadata=None, flush=False):
        if self.is_close:
            raise Exception("Dataset not built")

        if data == None:
            return

        data_ptr = self.dataset.tell()
        data_len = len(data)
        self.index.append([data_ptr, data_len, identity])
        self.dataset.write(data)

        # block align
        if self.align_fs_blocks:
            remainder = (data_ptr + data_len) % self.block_size
            if remainder != 0:
                self.dataset.write(bytearray(self.block_size - remainder))

        if self.metadata and metadata:
            self.metadata[identity] = metadata

        if flush:
            self.flush()

    def write_metadata(self, id, metadata):
        self.metadata[id] = metadata

    def flush_index(self, silent=False):
        print("index len", len(self.index))
        if not self.index and not silent:
            print("Warning: Index not built, couldn't flush")
            return

        with open(self.index_path_temp, 'wb') as f:
            pickle.dump(self.index, f)

        try:
            os.remove(self.index_path)
        except: pass
        os.rename(self.index_path_temp, self.index_path)
    
    def flush_metadata(self, silent=False):
        if not self.metadata and not silent:
            print("Warning: Metadata not built, couldn't flush")
            return

        with open(self.metadata_path_temp, 'wb') as f:
            pickle.dump(self.metadata, f)

        try:
            os.remove(self.metadata_path)
        except: pass
        os.rename(self.metadata_path_temp, self.metadata_path)

    def flush(self, silent=False):
        if not self.dataset and not silent:
            print("Warning: Dataset not built, couldn't flush")
            return

        self.dataset.flush()

    def close(self, silent=False):
        if not self.dataset and not silent:
            print("Warning: Dataset not built, couldn't flush")
            return

        #close the dataset filehandle and dump the pickle index
        self.flush()
        self.dataset.close()

if __name__ == "__main__":
    import argparse
    from dotmap import DotMap
    from omegaconf import OmegaConf
    print("Starting!")
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-c", "--config", type=str, default="/ssdwork/liling/sd-private/stable-diffusion-private/configs/based/animefull_bucket_test.yaml")
    args = argparser.parse_args()
    #read train config from yaml with OmegaConf
    train_config = OmegaConf.load(args.config)
    print(train_config)
    args = DotMap(train_config)

    bs = args["bs"]
    gas = args["gas"]
    # world_size = int(os.environ["WORLD_SIZE"])
    # rank = int(os.environ["LOCAL_RANK"])
    # global_rank = int(os.environ["RANK"])
    world_size, rank, global_rank = 1, 0, 0
    res_dropout = 0.0 if "res_dropout" not in args else args["res_dropout"]


    train_dataset = ShardedImageDataset(
        dataset_path=args["data_path"], 
        index_path=args["index_path"], 
        name="danbooru", shuffle=False, 
        bsz=bs*gas, threads=8, inner_transform=inner_transform, 
        world_size=world_size, local_rank=rank, global_rank=global_rank, 
        resolution_pkl=args["bucket_path"], max_size=(args["bucket_max_size_w"], args["bucket_max_size_h"]), 
        bucket_seed=args["bucket_seed"], res_dropout=res_dropout)
    train_dataset.shard(shuffle=True, epoch=args["epoch"], seed=args["seed"])
