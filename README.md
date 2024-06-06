# bend_vlm
Demo code for "Test-Time Debiasing of Vision-Language Embeddings".

## Preparing Datasets
Please download the [Celeba_HQ_dialog](https://github.com/ziqihuangg/CelebA-Dialog) and [FairFace](https://github.com/joojs/fairface) datasets from their sources.

Next, run the following to process the datasets (precompute the CLIP embeddings):

`python create_dataset.py --_MODEL_NAME 'clip-vit-base-patch16' --data_path '../../../gobi1/data/FairFace/' --meta_data_file_name 'fairface_label_train.csv' --dataset_name 'fairface'`

`python create_dataset.py --_MODEL_NAME 'clip-vit-base-patch32' --data_path '../../../gobi1/data/FairFace/' --meta_data_file_name 'fairface_label_train.csv' --dataset_name 'fairface'`

`python create_dataset.py --_MODEL_NAME 'clip-vit-base-patch32' --data_path '../../../gobi1/data/Celeba_HQ_dialog/' --meta_data_file_name 'CelebAMask-HQ-attribute-anno.txt' --dataset_name 'celeba'`

`python create_dataset.py --_MODEL_NAME 'clip-vit-base-patch32' --data_path '../../../gobi1/data/Celeba_HQ_dialog/' --meta_data_file_name 'CelebAMask-HQ-attribute-anno.txt' --dataset_name 'celeba'`


