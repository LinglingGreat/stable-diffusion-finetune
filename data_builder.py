import os
from ldm.data import dataset
import simplejpeg
import os.path as pt
import glob
import io
from PIL import Image
import numpy as np
import pickle

# data_dir = "/ssdwork/linzhihang/one_piece"
# output_path = "datasets/onepiece"
# name = "onepiece"

data_dir = "/ssdwork/liling/midjourney"
output_path = "datasets/midjourney"
name = "midjourney"

bucket_path = output_path + "/db_sizes.pkl"
tag_path = output_path + "/db_tags_ranked.pkl"
ext = ['png', 'jpg', 'jpeg', 'bmp']

builder = dataset.ImageDatasetBuilder(output_path, name, metadata=True)
builder.build()

bucket_data = {}
tag_data = {}
all_image = []
[all_image.extend(glob.glob(f'{data_dir}/img/' + '*.' + e)) for e in ext]
print(data_dir, len(all_image))

image_ids = list(range(len(all_image)))

for i, image_path in enumerate(all_image):
    img = Image.open(image_path)
    # data = Image.open(io.BytesIO(data))
    # mode = data.mode
    # data = np.asarray(data)
    # print(img.size, img.format, img.width, img.height)
    bucket_data[image_ids[i]] = [img.width, img.height]

    image_name = image_path[len(f'{data_dir}/img/'):]
    # print(image_name, image_name.split('.')[0])
    txt_path = os.path.join(data_dir, "txt", image_name.split('.')[0]+".txt")
    # print(txt_path)
    with open(txt_path, 'rt') as f:
        caption = f.read()
    caption = caption.strip('\n')
    tag_data[image_ids[i]] = caption

def encode_op(file_path):
    f = open(file_path, "rb")
    data = f.read()
    f.close()

    if simplejpeg.is_jpeg(data):
        try:
            simplejpeg.decode_jpeg(data)
        except Exception as e:
            print("jpeg", file_path, str(e))
            return None
    else:
        try:
            data = Image.open(io.BytesIO(data))
            data = data.convert("RGB")
            mode = data.mode
            data = np.asarray(data)
            data = simplejpeg.encode_jpeg(data, quality=91, colorspace=mode)
        except Exception as e:
            print("nojpeg", file_path, str(e))
            return None

    return data


builder.operate(encode_op, all_image, image_ids)
builder.flush()
builder.flush_index()
builder.flush_metadata()

with open(bucket_path, "wb") as f:
    pickle.dump(bucket_data, f)

with open(tag_path, "wb") as f:
    pickle.dump(tag_data, f)
