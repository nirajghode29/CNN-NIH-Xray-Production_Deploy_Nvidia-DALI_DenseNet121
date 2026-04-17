This project builds a deep learning pipeline for **multi-label classification** on the **NIH Chest X-ray dataset** using **PyTorch** and a **pretrained DenseNet121** backbone.

The main goal was to create a clean and practical training pipeline that covers:

- dataset preparation - Creating **DataPipeline**
- preprocessing and augmentation
- train/validation/test splitting
- **DataLoader Pipeline setup**
- class imbalance handling
- DenseNet121 fine-tuning for multi-label prediction

---

## 1. Project Overview

The NIH Chest X-ray dataset contains chest X-ray images where a single image can have **multiple disease findings at the same time**.

Because of that, this is a **multi-label classification problem**, not a single-label classification problem.

That design decision affected several parts of the pipeline:

- labels must be stored as **multi-hot vectors**
- the model output layer must produce one score per class
- the loss function must be **`BCEWithLogitsLoss`**
- prediction logic must use **`sigmoid` + threshold**
- class imbalance must be handled per class using **`pos_weight`**
- Increasing training efficiency using Torchvision v2, Nvidia DALI to shift image transforms and augmentation operations to GPU to avoid CPU bottle necking

---

## 2. Dataset Preparation

A custom PyTorch dataset class was used to load the NIH data.

### Dataset Pipeline

The custom `NIHDataset` class was designed to:

- read the NIH metadata CSV
- parse the `Finding Labels` column
- build a list of all unique classes
- create label-to-index mappings
- scan image folders and map image names to full file paths
- load each image from disk
- return an image and its corresponding target

### Key internal attributes

The dataset stores useful attributes such as:

- `all_labels` → list of all unique disease labels
- `class_to_idx` → mapping from class name to integer index
- `idx_to_class` → reverse mapping
- `image_paths` → mapping from image file name to its absolute path

This allowed us to later determine:

- number of classes
- class names
- class imbalance statistics

---

## 3. Label Processing

The NIH data is not a regular single-class dataset.

Each image may contain findings like:

- Atelectasis
- Infiltration
- Effusion
- Mass
- Pneumonia

and possibly more than one of them at once.

So instead of returning one integer class index, the dataset should return a **multi-hot encoded target vector**.

### Example

If we had 5 classes:

```python
["Atelectasis", "Cardiomegaly", "Effusion", "Mass", "Nodule"]



###Summary:

Without optimization:

[Training] Epoch 1:
	Step 64/1139 - Loss: 1.234 | Acc: 59.98% | Time Elapsed: 87.96Secs | Avg Time Per Batch: 1.37Secs
	Step 128/1139 - Loss: 1.169 | Acc: 61.74% | Time Elapsed: 179.53Secs | Avg Time Per Batch: 1.40Secs
	Step 192/1139 - Loss: 1.078 | Acc: 64.64% | Time Elapsed: 270.84Secs | Avg Time Per Batch: 1.41Secs
	Step 256/1139 - Loss: 1.024 | Acc: 67.52% | Time Elapsed: 364.17Secs | Avg Time Per Batch: 1.42Secs
	Step 320/1139 - Loss: 1.015 | Acc: 69.82% | Time Elapsed: 457.36Secs | Avg Time Per Batch: 1.43Secs
	Step 384/1139 - Loss: 0.998 | Acc: 70.61% | Time Elapsed: 551.99Secs | Avg Time Per Batch: 1.44Secs
	Step 448/1139 - Loss: 1.001 | Acc: 71.08% | Time Elapsed: 642.27Secs | Avg Time Per Batch: 1.43Secs
	Step 512/1139 - Loss: 1.053 | Acc: 70.20% | Time Elapsed: 732.40Secs | Avg Time Per Batch: 1.43Secs
	Step 576/1139 - Loss: 1.023 | Acc: 69.87% | Time Elapsed: 822.97Secs | Avg Time Per Batch: 1.43Secs
	Step 640/1139 - Loss: 1.038 | Acc: 69.38% | Time Elapsed: 917.52Secs | Avg Time Per Batch: 1.43Secs
	Step 704/1139 - Loss: 1.059 | Acc: 69.59% | Time Elapsed: 1015.34Secs | Avg Time Per Batch: 1.44Secs
	Step 768/1139 - Loss: 1.024 | Acc: 70.72% | Time Elapsed: 1114.64Secs | Avg Time Per Batch: 1.45Secs
	Step 832/1139 - Loss: 1.017 | Acc: 71.06% | Time Elapsed: 1214.79Secs | Avg Time Per Batch: 1.46Secs
	Step 896/1139 - Loss: 0.989 | Acc: 71.45% | Time Elapsed: 1320.34Secs | Avg Time Per Batch: 1.47Secs
	Step 960/1139 - Loss: 0.943 | Acc: 74.19% | Time Elapsed: 1426.48Secs | Avg Time Per Batch: 1.49Secs
	Step 1024/1139 - Loss: 0.968 | Acc: 73.17% | Time Elapsed: 1531.40Secs | Avg Time Per Batch: 1.50Secs
	Step 1088/1139 - Loss: 1.002 | Acc: 72.50% | Time Elapsed: 1633.61Secs | Avg Time Per Batch: 1.50Secs
	Step 1139/1139 - Loss: 0.778 | Acc: 73.20% | Time Elapsed: 1713.65Secs | Avg Time Per Batch: 1.50Secs
[Validation] Epoch 1:
	Validation Accuracy: 72.81%


With Optimization:
--batch-size 64 --num-workers 4 --epochs 1 --val-fraction 0.1 --test-fraction 0.1 --lr 0.001 --pretrained

[Training] Epoch 1:
        Step 64/1227 - Loss: 1.293 | Acc: 55.13% | Time Elapsed: 0.87Mins | Avg Time Per Image: 0.82Secs
        Step 128/1227 - Loss: 1.291 | Acc: 54.91% | Time Elapsed: 1.48Mins | Avg Time Per Image: 0.69Secs
        Step 192/1227 - Loss: 1.264 | Acc: 52.90% | Time Elapsed: 2.07Mins | Avg Time Per Image: 0.65Secs
        Step 256/1227 - Loss: 1.226 | Acc: 59.33% | Time Elapsed: 2.65Mins | Avg Time Per Image: 0.62Secs
        Step 320/1227 - Loss: 1.257 | Acc: 53.72% | Time Elapsed: 3.22Mins | Avg Time Per Image: 0.60Secs
        Step 384/1227 - Loss: 1.254 | Acc: 55.64% | Time Elapsed: 3.78Mins | Avg Time Per Image: 0.59Secs
        Step 448/1227 - Loss: 1.241 | Acc: 55.59% | Time Elapsed: 4.32Mins | Avg Time Per Image: 0.58Secs
        Step 512/1227 - Loss: 1.246 | Acc: 55.81% | Time Elapsed: 4.84Mins | Avg Time Per Image: 0.57Secs
        Step 576/1227 - Loss: 1.229 | Acc: 59.45% | Time Elapsed: 5.36Mins | Avg Time Per Image: 0.56Secs
        Step 640/1227 - Loss: 1.236 | Acc: 58.08% | Time Elapsed: 5.90Mins | Avg Time Per Image: 0.55Secs
        Step 704/1227 - Loss: 1.208 | Acc: 56.42% | Time Elapsed: 6.44Mins | Avg Time Per Image: 0.55Secs
        Step 768/1227 - Loss: 1.166 | Acc: 62.53% | Time Elapsed: 6.97Mins | Avg Time Per Image: 0.54Secs
        Step 832/1227 - Loss: 1.233 | Acc: 60.11% | Time Elapsed: 7.52Mins | Avg Time Per Image: 0.54Secs
        Step 896/1227 - Loss: 1.159 | Acc: 59.90% | Time Elapsed: 8.10Mins | Avg Time Per Image: 0.54Secs
        Step 960/1227 - Loss: 1.208 | Acc: 61.09% | Time Elapsed: 8.67Mins | Avg Time Per Image: 0.54Secs
        Step 1024/1227 - Loss: 1.232 | Acc: 54.92% | Time Elapsed: 9.22Mins | Avg Time Per Image: 0.54Secs
        Step 1088/1227 - Loss: 1.227 | Acc: 58.52% | Time Elapsed: 9.76Mins | Avg Time Per Image: 0.54Secs
        Step 1152/1227 - Loss: 1.206 | Acc: 57.20% | Time Elapsed: 10.33Mins | Avg Time Per Image: 0.54Secs
        Step 1216/1227 - Loss: 1.193 | Acc: 60.16% | Time Elapsed: 10.87Mins | Avg Time Per Image: 0.54Secs
        Step 1227/1227 - Loss: 1.173 | Acc: 62.31% | Time Elapsed: 10.96Mins | Avg Time Per Image: 0.54Secs
[Validation] Epoch 1:
        Validation Accuracy: 63.77%

Final summary
Train losses: [1.2295045789996102]
Validation accuracy: [63.76739207991437]

