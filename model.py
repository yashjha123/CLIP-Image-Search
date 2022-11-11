import numpy as np
# import pandas as pd
import glob

from tqdm import tqdm

import torch

input_text = "elephant drinking water"

import clip
from PIL import Image

import json

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} for torch")
print("Loading model...")
model, preprocess = clip.load("ViT-B/32", device=device)
print(f"Model Loaded")
path = "/home/test/Desktop/WebD/batch_images/batch_features/batch"
jsonFile = json.load(open("/home/test/Desktop/WebD/batch_images/compiled_data.json","r"))
print("Loaded data file")

def compute_txt(input_text):
    text = clip.tokenize([input_text]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)
        buffer2 = []
        all_records= []
        for txt in text_features: # Though, only one text
#             for i in tqdm(range(12)):
            for i in tqdm(range(12)):

                val = torch.load(open(f"{path}-{i}","rb"))

                t = txt.unsqueeze(1)
                v  = torch.matmul(val.to(torch.float32),t)
                buffer2.append(v)
            all_records.append(torch.cat(buffer2))
        return all_records
def compute(all_records):
    top_arrays = all_records[0].softmax(dim=0).cpu().topk(30,dim=0)
    top_vals = (top_arrays.values).squeeze(1).numpy()
    top_arrays = top_arrays.indices
    top_arrays = top_arrays.squeeze(1).numpy()
#     top_vals = all_records[0][top_arrays].softmax(dim=0).cpu().numpy()
    return top_arrays,top_vals
def get_image_url(top_arrays,top_vals):
    unsplash_bucket = "kds-9a6b71dbd7664edd75d2e747f6999eb31282c07d44de16cc0ec0696f/downloads/"
    flickr_bucket = "kds-0c8a46264ae2ccb5e293e650171a27f0bea5d40c4a610d3218154391/flickr30k_images/flickr30k_images/"
    #http://storage.googleapis.com/BUCKET_NAME/OBJECT_NAME
    d = {"unsplash-250x250":unsplash_bucket,"Flickr30k":flickr_bucket}
    ret = []
    for i, ind in enumerate(top_arrays):
        bucket = d[jsonFile[ind]["dataset"]]  
        file_name = "http://storage.googleapis.com/"+bucket+(jsonFile[ind]["id"])+".jpg"
        ret.append(file_name)
    return ret
    # for i,ind in enumerate(top_arrays):
    #     base_dir = d[jsonFile[ind]["dataset"]]
    #     file_name = base_dir+(jsonFile[ind]["id"])+".jpg"

    #     (cv2_imshow((file_name)))
    #     plt.grid(False)
    #     plt.axis('off')
    #     plt.title(f"Image {ind} [{(top_vals[i]*100):.2f}%]")
    #     plt.show() 
def allFunctions(txt):
    all_records = compute_txt(txt)
    top_arrays, top_vals = compute(all_records)
    return (get_image_url(top_arrays,top_vals))