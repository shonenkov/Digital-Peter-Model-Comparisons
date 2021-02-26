# Digital-Peter-Model-Comparisons

Neptune with experiments: https://ui.neptune.ai/aleksey.shonenkov/Digital-Peter-Model-Comparisons/

![](./pics/clovaai-v2.png)


## Getting Started
- Colab Usage: <a href="https://colab.research.google.com/github/shonenkov/Digital-Peter-Model-Comparisons/blob/master/jupyters/running_experiments.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

- Build image and run docker with jupyter environment:
```
docker build . --tag dpmc-image
docker run \
    --name dpmc-container \
    -p 7080:8888 \
    -e NVIDIA_VISIBLE_DEVICES=0 \
    dpmc-image
```
Go to https://127.0.0.1:7080, upload prepared notebook from `jupyters/running_experiments.ipynb` and run experiments

- For downloading prepared datasets (`peter|bentham|iam|saintgall`):
```
python scripts/download_dataset.py --dataset_name peter
```

## How to Run Experiments

### T1 None-VGG-None-CTC
```
python scripts/train.py \
  --checkpoint_path "" \
  --experiment_name "peter_T1" \
  --experiment_description "T1 None-VGG-None-CTC" \
  --dataset_name "peter" \
  --data_dir "../input" \
  --output_dir "../saved_models" \
  --image_w 1024 \
  --image_h 128 \
  --num_epochs 100 \
  --bs 16 \
  --num_workers 4 \
  --use_progress_bar 0 \
  --neptune_project "" \
  --neptune_token "" \
  --Transformation "None" \
  --FeatureExtraction "VGG" \
  --SequenceModeling "None" \
  --Prediction "CTC" \
  --seed 6955
```

### T2 None-ResNet-None-CTC
```
python scripts/train.py \
  --checkpoint_path "" \
  --experiment_name "peter_T2" \
  --experiment_description "T2 None-ResNet-None-CTC" \
  --dataset_name "peter" \
  --data_dir "../input" \
  --output_dir "../saved_models" \
  --image_w 1024 \
  --image_h 128 \
  --num_epochs 100 \
  --bs 16 \
  --num_workers 4 \
  --use_progress_bar 0 \
  --neptune_project "" \
  --neptune_token "" \
  --Transformation "None" \
  --FeatureExtraction "ResNet" \
  --SequenceModeling "None" \
  --Prediction "CTC" \
  --seed 6955
```

### T3 None-ResNet-BiLSTM-CTC
```
python scripts/train.py \
  --checkpoint_path "" \
  --experiment_name "peter_T3" \
  --experiment_description "T3 None-ResNet-BiLSTM-CTC" \
  --dataset_name "peter" \
  --data_dir "../input" \
  --output_dir "../saved_models" \
  --image_w 1024 \
  --image_h 128 \
  --num_epochs 100 \
  --bs 16 \
  --num_workers 4 \
  --use_progress_bar 0 \
  --neptune_project "" \
  --neptune_token "" \
  --Transformation "None" \
  --FeatureExtraction "ResNet" \
  --SequenceModeling "BiLSTM" \
  --Prediction "CTC" \
  --seed 6955
```

### P1 None-RCNN-None-CTC

```
python scripts/train.py \
  --checkpoint_path "" \
  --experiment_name "peter_P1" \
  --experiment_description "P1 None-RCNN-None-CTC" \
  --dataset_name "peter" \
  --data_dir "../input" \
  --output_dir "../saved_models" \
  --image_w 1024 \
  --image_h 128 \
  --num_epochs 100 \
  --bs 16 \
  --num_workers 4 \
  --use_progress_bar 0 \
  --neptune_project "" \
  --neptune_token "" \
  --Transformation "None" \
  --FeatureExtraction "RCNN" \
  --SequenceModeling "None" \
  --Prediction "CTC" \
  --seed 6955
```

### CRNN: None-VGG-BiLSTM-CTC
```
python scripts/train.py \
  --checkpoint_path "" \
  --experiment_name "peter_CRNN" \
  --experiment_description "CRNN None-VGG-BiLSTM-CTC" \
  --dataset_name "peter" \
  --data_dir "../input" \
  --output_dir "../saved_models" \
  --image_w 1024 \
  --image_h 128 \
  --num_epochs 100 \
  --bs 16 \
  --num_workers 4 \
  --use_progress_bar 0 \
  --neptune_project "" \
  --neptune_token "" \
  --Transformation "None" \
  --FeatureExtraction "VGG" \
  --SequenceModeling "BiLSTM" \
  --Prediction "CTC" \
  --seed 6955
```

### GRCNN: None-RCNN-BiLSTM-CTC
```
python scripts/train.py \
  --checkpoint_path "" \
  --experiment_name "peter_GRCNN" \
  --experiment_description "GRCNN None-RCNN-BiLSTM-CTC" \
  --dataset_name "peter" \
  --data_dir "../input" \
  --output_dir "../saved_models" \
  --image_w 1024 \
  --image_h 128 \
  --num_epochs 100 \
  --bs 16 \
  --num_workers 4 \
  --use_progress_bar 0 \
  --neptune_project "" \
  --neptune_token "" \
  --Transformation "None" \
  --FeatureExtraction "RCNN" \
  --SequenceModeling "BiLSTM" \
  --Prediction "CTC" \
  --seed 6955
```

### R2AM: None-RCNN-None-Attn
```
python scripts/train.py \
  --checkpoint_path "" \
  --experiment_name "peter_R2AM" \
  --experiment_description "R2AM None-RCNN-None-Attn" \
  --dataset_name "peter" \
  --data_dir "../input" \
  --output_dir "../saved_models" \
  --image_w 1024 \
  --image_h 128 \
  --num_epochs 100 \
  --bs 16 \
  --num_workers 4 \
  --use_progress_bar 0 \
  --neptune_project "" \
  --neptune_token "" \
  --Transformation "None" \
  --FeatureExtraction "RCNN" \
  --SequenceModeling "None" \
  --Prediction "Attn" \
  --seed 6955
```

### STAR-Net TPS-ResNet-BiLSTM-CTC
```
python scripts/train.py \
  --checkpoint_path "" \
  --experiment_name "peter_STARNET" \
  --experiment_description "STAR-Net TPS-ResNet-BiLSTM-CTC" \
  --dataset_name "peter" \
  --data_dir "../input" \
  --output_dir "../saved_models" \
  --image_w 1024 \
  --image_h 128 \
  --num_epochs 100 \
  --bs 16 \
  --num_workers 4 \
  --use_progress_bar 0 \
  --neptune_project "" \
  --neptune_token "" \
  --Transformation "TPS" \
  --FeatureExtraction "ResNet" \
  --SequenceModeling "BiLSTM" \
  --Prediction "CTC" \
  --seed 6955
```

## Acknowledgements

This implementation has been based on repository [deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark)