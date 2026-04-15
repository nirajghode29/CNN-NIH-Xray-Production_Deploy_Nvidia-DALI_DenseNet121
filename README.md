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
