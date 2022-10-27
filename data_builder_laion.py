import os
from ldm.data import dataset
import simplejpeg
import os.path as pt
import glob
import io
from PIL import Image
import numpy as np
import pickle
import json

data_dir = "/data/liling_ssdwork/laion2B-en/laion-1024"
output_path = "datasets/laionart_special_size"
name = "laionartsp"

bucket_path = output_path + "/db_sizes.pkl"
tag_path = output_path + "/db_tags_ranked.pkl"
ext = ['png', 'jpg', 'jpeg', 'bmp']

builder = dataset.ImageDatasetBuilder(output_path, name, metadata=True)
builder.build()

bucket_data = {}
tag_data = {}
all_image = []
bucket_dict = {}
for i in range(487):
    folder = '0'*(5-len(str(i)))+str(i)
    folder_path = os.path.join(data_dir, folder)
    all_json = glob.glob(f'{data_dir}/{folder}/' + '*.json')
    print(i, len(all_json))
    for j in all_json:
        with open(j) as f:
            js_data = json.load(f)
            if js_data["LANGUAGE"] == "en":
                image_path = j[:-5] + ".jpg"
                img = Image.open(image_path)
                k = str(img.width)+"_"+str(img.height)
                bucket_dict[k] = bucket_dict.get(k, 0) + 1
                # if bucket_dict[k] >= 4000:
                #     continue
                if k in ["576_1024", "1024_576", "512_512"]:
                    all_image.append(image_path)
print(bucket_dict["576_1024"], bucket_dict["1024_576"], bucket_dict["512_512"])
print(data_dir, len(all_image))

image_ids = list(range(len(all_image)))

for i, image_path in enumerate(all_image):
    img = Image.open(image_path)
    # data = Image.open(io.BytesIO(data))
    # mode = data.mode
    # data = np.asarray(data)
    # print(img.size, img.format, img.width, img.height)
    bucket_data[image_ids[i]] = [img.width, img.height]

    txt_path = image_path[:-4] + ".txt"
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
