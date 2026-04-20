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



###Model Comparison: Filter out models with accuracy on 5 Epochs.

Comparison Between RestNet and MobileNet:
Model: resnet18
Train losses: [1.0415923982591482, 0.9263832987374956, 0.8471168558968697, 0.7686603944672737, 0.6759863143396649]
Validation accuracy: [71.58441352519127, 71.4734213342847, 76.83156935029928, 78.69227415071154, 79.95481032227374]
Test accuracy: 79.92%
----------------------------------------
Model: mobilenet_v2
Train losses: [1.0668273168360205, 0.9357404465974591, 0.8636343249870493, 0.8025839413290793, 0.7568629685707372]
Validation accuracy: [70.04915368454434, 74.85194434534428, 76.65913505371229, 78.03107781345385, 77.60375787846355]
Test accuracy: 77.65%

Model: vgg16
Train losses: [1.2481776367092676, 1.1596507639931584, 1.10912646951963, 1.0740565112575633, 1.0278751656789553]
Validation accuracy: [53.75074325127839, 65.11079399056567, 70.6267094779403, 69.7883220359139, 75.65386292464423]
Test accuracy: 75.54%
----------------------------------------
Model: densenet121
Train losses: [1.0317207735150251, 0.9203205055693057, 0.8477077343741761, 0.7925363405020915, 0.7338018331376148]
Validation accuracy: [73.37812661037776, 74.20184722717723, 75.41760811828595, 76.06572323304395, 78.84330281048084]
Test accuracy: 78.76%
----------------------------------------

