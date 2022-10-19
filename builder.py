from basedformer import dataset
import simplejpeg
import os.path as pt
import glob
import io
from PIL import Image
import numpy as np
import pickle

image_dir = "/ssdwork/liling/sd-private/outputs"
output_path = "/ssdwork/liling/sd-private/stable-diffusion-private/danbooru"
name = "e621"
bucket_path = output_path + "/e621_sizes.pkl"
ext = ['png', 'jpg', 'jpeg', 'bmp']

builder = dataset.ImageDatasetBuilder(output_path, name, metadata=True)
builder.build()

bucket_data = {}
all_image = []
# img_path = Path(f'{self.root_dir}/img/')
# [self.paths.extend(list(img_path.rglob(f"*.{e}"))) for e in ext]
[all_image.extend(glob.glob(f'{image_dir}' + '*.' + e)) for e in ext]
print(image_dir, len(all_image))

image_ids = list(range(len(all_image)))

for i, image_path in enumerate(all_image):
    img = Image.open(image_path)
    print(img.size, img.format, img.width, img.height)
    bucket_data[image_ids[i]] = [img.width, img.height]

def encode_op(file_path):
    f = open(file_path, "rb")
    data = f.read()
    f.close()

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


builder.operate(encode_op, all_image, image_ids)
builder.flush()
builder.flush_index()
builder.flush_metadata()

with open(bucket_path, "wb") as f:
    pickle.dump(bucket_data, f)
