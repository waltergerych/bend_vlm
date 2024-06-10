import torch
from transformers import CLIPModel, AutoProcessor
import pandas as pd
from matplotlib.image import imread
import json
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Argument Parser")
parser.add_argument('--_MODEL_NAME', type=str, 
                    default='clip-vit-base-patch16') #clip model to use
parser.add_argument('--data_path', type=str, 
                    default='Celeba_HQ_dialog/') #path to celeba or fairface data
parser.add_argument('--meta_data_file_name', type=str, 
                    default='CelebAMask-HQ-attribute-anno.txt') 
parser.add_argument('--dataset_name', type=str, 
                    default='celeba') 
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_MODEL_NAME = args._MODEL_NAME 
data_path = args.data_path 
meta_data_file_name = args.meta_data_file_name 
dataset_name = args.dataset_name

model_ID = f'openai/{_MODEL_NAME}' 
vl_model = CLIPModel.from_pretrained(model_ID).to(device)
processor = AutoProcessor.from_pretrained(model_ID)

def get_celeba_metadata(data_path, meta_data_file_name):
    def add_path(row):
        return 'image/' + row

    img_files = pd.read_csv(data_path + meta_data_file_name, skiprows=1, sep=" ")
    img_files.index = img_files.index.get_level_values(0)
    img_files = img_files.reset_index()
    img_files = img_files.rename(columns = {'index': 'filename'})
    img_files['file'] = img_files['filename'].apply(add_path)
    feature_names = img_files.columns
    return img_files, feature_names

def get_fairface_metadata(data_path, meta_data_file_name):
    def add_path(row):
        return data_path + row

    img_files = pd.read_csv(data_path + meta_data_file_name)
    img_files['face_name_align'] = img_files['file'].apply(add_path)
    img_files.head()

    def strip_filename(row):
        return row.split("/")[-1]
    img_files['filename'] = img_files['face_name_align']
    img_files['filename'] = img_files['filename'].apply(strip_filename)
    img_files.head()

    feature_names = ['age', 'race', 'gender', 'file', 'filename']
    meta_df = img_files[feature_names]

    return meta_df, feature_names

if dataset_name == 'celeba':
    meta_df, feature_names = get_celeba_metadata(data_path, meta_data_file_name)
    meta_df.head()

elif dataset_name == 'fairface':
    meta_df, feature_names = get_fairface_metadata(data_path, meta_data_file_name)
    meta_df.head()

else:
    print("Dataset not defined")

batch_size = 256

indx = 0

embed_list = []

while indx < len(meta_df):
    batch_files = list(meta_df[indx:indx+batch_size]['file'])
    batch_imgs = [imread(data_path + x) for x in batch_files]

    with torch.no_grad():
        inputs = processor(images=batch_imgs, return_tensors="pt").to(device)

        image_features = vl_model.get_image_features(**inputs).to('cpu').numpy()
    
    im_file_names = [bf.split(data_path)[-1] for bf in batch_files]
    for i_indx, i in enumerate(im_file_names):
        embed_dict = {}
        embed_dict['embedding'] = [image_features[i_indx]]
        feats = meta_df[meta_df['file']==i] 
        for feature in feature_names:
            embed_dict[feature] = feats[feature].item()
        embed_list.append(embed_dict)
                
    print(f'{indx}/{len(meta_df)}')
    indx += batch_size

    
    
for i in range(len(embed_list)):
    embed_list[i]['embedding'][0] = [f.astype(float) for f in embed_list[i]['embedding'][0]]

with open(f'data/{dataset_name}_featurized_{_MODEL_NAME}.jsonl', 'w') as fout:
    for e_dict in embed_list:
        json.dump(e_dict, fout)

np.random.seed(42) 
fold_dict = {}
for i in range(5):
    fold_dict[i] = {}
    indices = np.arange(len(embed_list))
    np.random.shuffle(indices)

    split_point = int(len(indices) * (1 - 0.5))

    train_indices = indices[:split_point]
    test_indices = indices[split_point:]

    fold_dict[i]['train_indices'] = [int(x) for x in train_indices]
    fold_dict[i]['test_indices'] = [int(x) for x in test_indices]

with open(f'data/fold_indices/{dataset_name}_featurized_{_MODEL_NAME}_folds.jsonl', 'w') as fout:
    json.dump(fold_dict, fout)
