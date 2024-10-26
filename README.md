# "BendVLM: Test-Time Debiasing of Vision-Language Embeddings" (NeurIPS 2024)
Demo code for "BendVLM: Test-Time Debiasing of Vision-Language Embeddings" (NeurIPS 2024).

## Installing Packages

```console
pip install -r requirements.txt
```

## Preparing Datasets
Please download the [Celeba_HQ_dialog](https://github.com/ziqihuangg/CelebA-Dialog) and [FairFace](https://github.com/joojs/fairface) datasets from their sources.

Next, run the following to process the datasets (precompute the CLIP embeddings):

```console
python create_dataset.py --_MODEL_NAME 'clip-vit-base-patch16' --data_path 'PATH_TO_FAIRFACE' --meta_data_file_name 'fairface_label_train.csv' --dataset_name 'fairface'
python create_dataset.py --_MODEL_NAME 'clip-vit-large-patch14' --data_path 'PATH_TO_FAIRFACE'  --meta_data_file_name 'fairface_label_train.csv' --dataset_name 'fairface'
python create_dataset.py --_MODEL_NAME 'clip-vit-base-patch16' --data_path 'PATH_TO_CELEBA'  --meta_data_file_name 'CelebAMask-HQ-attribute-anno.txt' --dataset_name 'celeba'
python create_dataset.py --_MODEL_NAME 'clip-vit-large-patch14' --data_path 'PATH_TO_CELEBA' --meta_data_file_name 'CelebAMask-HQ-attribute-anno.txt' --dataset_name 'celeba'
```

## Pre-Generating Query Augmentations

Rather than generating gender and race augmentations on-the-fly using our AttributeAugment componenent, for the purposes of our experiments only we pre-generate the augmenations for efficiency. You can pre-generate these augmentations by running:

```console
python get_pregenerated_attribute_augmentations.py --query_type "hair" --att_to_debias "gender"
python get_pregenerated_attribute_augmentations.py --query_type "stereotype" --att_to_debias "gender"
python get_pregenerated_attribute_augmentations.py --query_type "stereotype" --att_to_debias "race"
```
## Experimental Results

You can perform a comparative analysis between Bend_VLM and the compared methods by running the cells in the `demo_debias.ipynb` notebook. You can change the dataset, protected atttribute (race/gender), query class type (hair or stereotype), and CLIP embedding model by passing in the appropriate config file in `config = yaml.safe_load(open("experimental_configs/celeba_hair_gender_clip-vit-base-patch16.yml"))`. Each config file is named as `{dataset}_{protected_attribute}_{query_type}_{model_name}`.

## Citation

```
@inproceedings{gerych2024bendvlm,
 title={BendVLM: Test Time Debiasing Of Pretrained Vision-Language Models},
 author={Walter Gerych and Haoran Zhang and Kimia Hamidieh and Eileen Pan and Maanas Sharma and Thomas Hartvigsen and Marzyeh Ghassemi},
 booktitle = {Advances in Neural Information Processing Systems},
 year = {2024}
}
```
