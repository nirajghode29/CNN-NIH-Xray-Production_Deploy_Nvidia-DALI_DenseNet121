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
- NVIDIA DALI improved image transform, augmentation and normalization speed by upto 55% and improved training speed.

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


###Final Model Selection: DenseNet121:

Torch CPU threads: 1
DALI threads: 4
Dataset size: 112120
Num classes: 15
Train samples: 78484
Val samples: 16818
Test samples: 16818
[Training] Epoch 1:
        Step 96/818 - Loss: 1.203 | Acc: 56.68% | Time Elapsed: 0.73Mins | Avg Time Per Batch: 0.46Secs
        Step 192/818 - Loss: 1.070 | Acc: 67.49% | Time Elapsed: 1.43Mins | Avg Time Per Batch: 0.45Secs
        Step 288/818 - Loss: 1.079 | Acc: 67.91% | Time Elapsed: 2.12Mins | Avg Time Per Batch: 0.44Secs
        Step 384/818 - Loss: 1.031 | Acc: 69.21% | Time Elapsed: 2.80Mins | Avg Time Per Batch: 0.44Secs
        Step 480/818 - Loss: 1.018 | Acc: 70.14% | Time Elapsed: 3.49Mins | Avg Time Per Batch: 0.44Secs
libpng warning: iCCP: profile 'ICC Profile': 'GRAY': Gray color space not permitted on RGB PNG
        Step 576/818 - Loss: 0.984 | Acc: 71.49% | Time Elapsed: 4.18Mins | Avg Time Per Batch: 0.44Secs
libpng warning: iCCP: profile 'ICC Profile': 'GRAY': Gray color space not permitted on RGB PNG
        Step 672/818 - Loss: 0.969 | Acc: 72.41% | Time Elapsed: 4.86Mins | Avg Time Per Batch: 0.43Secs
        Step 768/818 - Loss: 0.969 | Acc: 72.92% | Time Elapsed: 5.55Mins | Avg Time Per Batch: 0.43Secs
        Step 818/818 - Loss: 0.939 | Acc: 72.90% | Time Elapsed: 5.92Mins | Avg Time Per Batch: 0.43Secs
[Validation] Epoch 1:
        Validation Accuracy: 75.68%
[Training] Epoch 2:
        Step 96/818 - Loss: 0.925 | Acc: 74.54% | Time Elapsed: 0.68Mins | Avg Time Per Batch: 0.43Secs
        Step 192/818 - Loss: 0.911 | Acc: 74.28% | Time Elapsed: 1.36Mins | Avg Time Per Batch: 0.43Secs
        Step 288/818 - Loss: 0.891 | Acc: 74.75% | Time Elapsed: 2.05Mins | Avg Time Per Batch: 0.43Secs
        Step 384/818 - Loss: 0.905 | Acc: 75.03% | Time Elapsed: 2.73Mins | Avg Time Per Batch: 0.43Secs
        Step 480/818 - Loss: 0.921 | Acc: 75.76% | Time Elapsed: 3.42Mins | Avg Time Per Batch: 0.43Secs
libpng warning: iCCP: profile 'ICC Profile': 'GRAY': Gray color space not permitted on RGB PNG
        Step 576/818 - Loss: 0.912 | Acc: 75.40% | Time Elapsed: 4.10Mins | Avg Time Per Batch: 0.43Secs
        Step 672/818 - Loss: 0.894 | Acc: 75.21% | Time Elapsed: 4.79Mins | Avg Time Per Batch: 0.43Secs
libpng warning: iCCP: profile 'ICC Profile': 'GRAY': Gray color space not permitted on RGB PNG
        Step 768/818 - Loss: 0.896 | Acc: 75.93% | Time Elapsed: 5.47Mins | Avg Time Per Batch: 0.43Secs
        Step 818/818 - Loss: 0.925 | Acc: 74.36% | Time Elapsed: 5.84Mins | Avg Time Per Batch: 0.43Secs
[Validation] Epoch 2:
        Validation Accuracy: 77.09%
[Training] Epoch 3:
libpng warning: iCCP: profile 'ICC Profile': 'GRAY': Gray color space not permitted on RGB PNG
        Step 96/818 - Loss: 0.830 | Acc: 76.81% | Time Elapsed: 0.68Mins | Avg Time Per Batch: 0.43Secs
        Step 192/818 - Loss: 0.810 | Acc: 78.26% | Time Elapsed: 1.35Mins | Avg Time Per Batch: 0.42Secs
        Step 288/818 - Loss: 0.843 | Acc: 76.85% | Time Elapsed: 2.03Mins | Avg Time Per Batch: 0.42Secs
libpng warning: iCCP: profile 'ICC Profile': 'GRAY': Gray color space not permitted on RGB PNG
        Step 384/818 - Loss: 0.828 | Acc: 77.16% | Time Elapsed: 2.73Mins | Avg Time Per Batch: 0.43Secs
        Step 480/818 - Loss: 0.854 | Acc: 77.05% | Time Elapsed: 7.40Mins | Avg Time Per Batch: 0.93Secs
        Step 576/818 - Loss: 0.818 | Acc: 77.64% | Time Elapsed: 8.22Mins | Avg Time Per Batch: 0.86Secs
        Step 672/818 - Loss: 0.840 | Acc: 77.42% | Time Elapsed: 9.14Mins | Avg Time Per Batch: 0.82Secs
libpng warning: iCCP: profile 'ICC Profile': 'GRAY': Gray color space not permitted on RGB PNG
        Step 768/818 - Loss: 0.835 | Acc: 77.19% | Time Elapsed: 9.85Mins | Avg Time Per Batch: 0.77Secs
        Step 818/818 - Loss: 0.835 | Acc: 77.72% | Time Elapsed: 10.22Mins | Avg Time Per Batch: 0.75Secs
[Validation] Epoch 3:
        Validation Accuracy: 76.13%
[Training] Epoch 4:
libpng warning: iCCP: profile 'ICC Profile': 'GRAY': Gray color space not permitted on RGB PNG
        Step 96/818 - Loss: 0.752 | Acc: 79.16% | Time Elapsed: 0.68Mins | Avg Time Per Batch: 0.43Secs
        Step 192/818 - Loss: 0.760 | Acc: 79.47% | Time Elapsed: 1.35Mins | Avg Time Per Batch: 0.42Secs
        Step 288/818 - Loss: 0.771 | Acc: 79.00% | Time Elapsed: 2.05Mins | Avg Time Per Batch: 0.43Secs
libpng warning: iCCP: profile 'ICC Profile': 'GRAY': Gray color space not permitted on RGB PNG
        Step 384/818 - Loss: 0.766 | Acc: 78.84% | Time Elapsed: 2.77Mins | Avg Time Per Batch: 0.43Secs
        Step 480/818 - Loss: 0.768 | Acc: 78.83% | Time Elapsed: 3.46Mins | Avg Time Per Batch: 0.43Secs
        Step 576/818 - Loss: 0.775 | Acc: 79.00% | Time Elapsed: 4.15Mins | Avg Time Per Batch: 0.43Secs
        Step 672/818 - Loss: 0.779 | Acc: 79.14% | Time Elapsed: 4.84Mins | Avg Time Per Batch: 0.43Secs
        Step 768/818 - Loss: 0.807 | Acc: 77.78% | Time Elapsed: 5.53Mins | Avg Time Per Batch: 0.43Secs
        Step 818/818 - Loss: 0.800 | Acc: 78.29% | Time Elapsed: 5.87Mins | Avg Time Per Batch: 0.43Secs
[Validation] Epoch 4:
        Validation Accuracy: 80.29%
[Training] Epoch 5:
libpng warning: iCCP: profile 'ICC Profile': 'GRAY': Gray color space not permitted on RGB PNG
        Step 96/818 - Loss: 0.692 | Acc: 80.78% | Time Elapsed: 0.69Mins | Avg Time Per Batch: 0.43Secs
        Step 192/818 - Loss: 0.672 | Acc: 82.08% | Time Elapsed: 1.39Mins | Avg Time Per Batch: 0.43Secs
        Step 288/818 - Loss: 0.700 | Acc: 80.82% | Time Elapsed: 2.06Mins | Avg Time Per Batch: 0.43Secs
        Step 384/818 - Loss: 0.697 | Acc: 80.84% | Time Elapsed: 2.73Mins | Avg Time Per Batch: 0.43Secs
        Step 480/818 - Loss: 0.700 | Acc: 80.69% | Time Elapsed: 3.41Mins | Avg Time Per Batch: 0.43Secs
        Step 576/818 - Loss: 0.700 | Acc: 81.00% | Time Elapsed: 4.11Mins | Avg Time Per Batch: 0.43Secs
        Step 672/818 - Loss: 0.690 | Acc: 80.69% | Time Elapsed: 4.78Mins | Avg Time Per Batch: 0.43Secs
        Step 768/818 - Loss: 0.762 | Acc: 79.23% | Time Elapsed: 5.45Mins | Avg Time Per Batch: 0.43Secs
libpng warning: iCCP: profile 'ICC Profile': 'GRAY': Gray color space not permitted on RGB PNG
        Step 818/818 - Loss: 0.713 | Acc: 79.90% | Time Elapsed: 5.81Mins | Avg Time Per Batch: 0.43Secs
[Validation] Epoch 5:
        Validation Accuracy: 79.23%
[Training] Epoch 6:
        Step 96/818 - Loss: 0.622 | Acc: 83.01% | Time Elapsed: 0.68Mins | Avg Time Per Batch: 0.42Secs
        Step 192/818 - Loss: 0.612 | Acc: 82.99% | Time Elapsed: 1.35Mins | Avg Time Per Batch: 0.42Secs
        Step 288/818 - Loss: 0.612 | Acc: 83.14% | Time Elapsed: 2.02Mins | Avg Time Per Batch: 0.42Secs
        Step 384/818 - Loss: 0.629 | Acc: 82.69% | Time Elapsed: 2.71Mins | Avg Time Per Batch: 0.42Secs
        Step 480/818 - Loss: 0.631 | Acc: 83.04% | Time Elapsed: 3.39Mins | Avg Time Per Batch: 0.42Secs
        Step 576/818 - Loss: 0.719 | Acc: 80.76% | Time Elapsed: 4.07Mins | Avg Time Per Batch: 0.42Secs
libpng warning: iCCP: profile 'ICC Profile': 'GRAY': Gray color space not permitted on RGB PNG
        Step 672/818 - Loss: 0.711 | Acc: 80.61% | Time Elapsed: 4.75Mins | Avg Time Per Batch: 0.42Secs
libpng warning: iCCP: profile 'ICC Profile': 'GRAY': Gray color space not permitted on RGB PNG
        Step 768/818 - Loss: 0.675 | Acc: 81.23% | Time Elapsed: 5.42Mins | Avg Time Per Batch: 0.42Secs
        Step 818/818 - Loss: 0.668 | Acc: 81.35% | Time Elapsed: 5.79Mins | Avg Time Per Batch: 0.42Secs
[Validation] Epoch 6:
        Validation Accuracy: 81.55%
[Training] Epoch 7:
        Step 96/818 - Loss: 0.569 | Acc: 84.03% | Time Elapsed: 0.68Mins | Avg Time Per Batch: 0.42Secs
        Step 192/818 - Loss: 0.564 | Acc: 84.24% | Time Elapsed: 1.35Mins | Avg Time Per Batch: 0.42Secs
libpng warning: iCCP: profile 'ICC Profile': 'GRAY': Gray color space not permitted on RGB PNG
        Step 288/818 - Loss: 0.551 | Acc: 84.53% | Time Elapsed: 2.05Mins | Avg Time Per Batch: 0.43Secs
        Step 384/818 - Loss: 0.541 | Acc: 85.09% | Time Elapsed: 2.72Mins | Avg Time Per Batch: 0.42Secs
libpng warning: iCCP: profile 'ICC Profile': 'GRAY': Gray color space not permitted on RGB PNG
        Step 480/818 - Loss: 0.564 | Acc: 84.36% | Time Elapsed: 3.39Mins | Avg Time Per Batch: 0.42Secs
        Step 576/818 - Loss: 0.566 | Acc: 84.23% | Time Elapsed: 4.06Mins | Avg Time Per Batch: 0.42Secs
        Step 672/818 - Loss: 0.572 | Acc: 84.13% | Time Elapsed: 4.76Mins | Avg Time Per Batch: 0.42Secs
        Step 768/818 - Loss: 0.594 | Acc: 83.73% | Time Elapsed: 5.44Mins | Avg Time Per Batch: 0.42Secs
        Step 818/818 - Loss: 0.576 | Acc: 83.63% | Time Elapsed: 5.77Mins | Avg Time Per Batch: 0.42Secs
[Validation] Epoch 7:
        Validation Accuracy: 82.36%
[Training] Epoch 8:
        Step 96/818 - Loss: 0.487 | Acc: 86.29% | Time Elapsed: 0.70Mins | Avg Time Per Batch: 0.44Secs
        Step 192/818 - Loss: 0.469 | Acc: 86.88% | Time Elapsed: 1.39Mins | Avg Time Per Batch: 0.43Secs
libpng warning: iCCP: profile 'ICC Profile': 'GRAY': Gray color space not permitted on RGB PNG
        Step 288/818 - Loss: 0.489 | Acc: 86.37% | Time Elapsed: 2.07Mins | Avg Time Per Batch: 0.43Secs
        Step 384/818 - Loss: 0.485 | Acc: 86.33% | Time Elapsed: 2.75Mins | Avg Time Per Batch: 0.43Secs
        Step 480/818 - Loss: 0.487 | Acc: 86.51% | Time Elapsed: 3.47Mins | Avg Time Per Batch: 0.43Secs
        Step 576/818 - Loss: 0.518 | Acc: 85.76% | Time Elapsed: 4.16Mins | Avg Time Per Batch: 0.43Secs
        Step 672/818 - Loss: 0.519 | Acc: 85.28% | Time Elapsed: 4.84Mins | Avg Time Per Batch: 0.43Secs
libpng warning: iCCP: profile 'ICC Profile': 'GRAY': Gray color space not permitted on RGB PNG
        Step 768/818 - Loss: 0.507 | Acc: 85.78% | Time Elapsed: 5.53Mins | Avg Time Per Batch: 0.43Secs
        Step 818/818 - Loss: 0.552 | Acc: 85.16% | Time Elapsed: 5.89Mins | Avg Time Per Batch: 0.43Secs
[Validation] Epoch 8:
        Validation Accuracy: 84.75%
[Training] Epoch 9:
        Step 96/818 - Loss: 0.435 | Acc: 87.47% | Time Elapsed: 0.67Mins | Avg Time Per Batch: 0.42Secs
        Step 192/818 - Loss: 0.428 | Acc: 88.21% | Time Elapsed: 1.35Mins | Avg Time Per Batch: 0.42Secs
libpng warning: iCCP: profile 'ICC Profile': 'GRAY': Gray color space not permitted on RGB PNG
        Step 288/818 - Loss: 0.425 | Acc: 87.89% | Time Elapsed: 2.05Mins | Avg Time Per Batch: 0.43Secs
        Step 384/818 - Loss: 0.453 | Acc: 87.45% | Time Elapsed: 2.73Mins | Avg Time Per Batch: 0.43Secs
        Step 480/818 - Loss: 0.449 | Acc: 87.37% | Time Elapsed: 3.41Mins | Avg Time Per Batch: 0.43Secs
        Step 576/818 - Loss: 0.463 | Acc: 86.95% | Time Elapsed: 4.09Mins | Avg Time Per Batch: 0.43Secs
        Step 672/818 - Loss: 0.454 | Acc: 87.15% | Time Elapsed: 4.76Mins | Avg Time Per Batch: 0.43Secs
        Step 768/818 - Loss: 0.474 | Acc: 86.72% | Time Elapsed: 5.46Mins | Avg Time Per Batch: 0.43Secs
libpng warning: iCCP: profile 'ICC Profile': 'GRAY': Gray color space not permitted on RGB PNG
        Step 818/818 - Loss: 0.461 | Acc: 87.22% | Time Elapsed: 5.80Mins | Avg Time Per Batch: 0.43Secs
[Validation] Epoch 9:
        Validation Accuracy: 86.83%
[Training] Epoch 10:
        Step 96/818 - Loss: 0.387 | Acc: 88.87% | Time Elapsed: 0.67Mins | Avg Time Per Batch: 0.42Secs
        Step 192/818 - Loss: 0.384 | Acc: 89.34% | Time Elapsed: 1.37Mins | Avg Time Per Batch: 0.43Secs
        Step 288/818 - Loss: 0.385 | Acc: 89.01% | Time Elapsed: 2.04Mins | Avg Time Per Batch: 0.43Secs
        Step 384/818 - Loss: 0.404 | Acc: 88.57% | Time Elapsed: 2.72Mins | Avg Time Per Batch: 0.42Secs
        Step 480/818 - Loss: 0.409 | Acc: 88.68% | Time Elapsed: 3.39Mins | Avg Time Per Batch: 0.42Secs
        Step 576/818 - Loss: 0.402 | Acc: 88.62% | Time Elapsed: 4.09Mins | Avg Time Per Batch: 0.43Secs
libpng warning: iCCP: profile 'ICC Profile': 'GRAY': Gray color space not permitted on RGB PNG
        Step 672/818 - Loss: 0.413 | Acc: 88.27% | Time Elapsed: 4.76Mins | Avg Time Per Batch: 0.43Secs
libpng warning: iCCP: profile 'ICC Profile': 'GRAY': Gray color space not permitted on RGB PNG
        Step 768/818 - Loss: 0.430 | Acc: 87.98% | Time Elapsed: 5.44Mins | Avg Time Per Batch: 0.42Secs
        Step 818/818 - Loss: 0.423 | Acc: 87.85% | Time Elapsed: 5.80Mins | Avg Time Per Batch: 0.43Secs
[Validation] Epoch 10:
        Validation Accuracy: 85.53%

Final summary
Train losses: [1.0365152495997458, 0.9103261829879873, 0.8344724423873687, 0.7758005154307722, 0.7038509688342405, 0.6541387189571315, 0.5674298937571369, 0.5000766215158267, 0.4498674208639886, 0.40396104330392807]
Validation accuracy: [75.67548500881834, 77.08563590045071, 76.12972761120909, 80.28728199098569, 79.22908093278464, 81.54732510288066, 82.36057221242406, 84.75485008818342, 86.83127572016461, 85.52968841857731]

[Test evaluation]
        Test Accuracy: 85.11%


Results:
<img width="869" height="734" alt="image" src="https://github.com/user-attachments/assets/17f4c0c1-9933-4a24-a253-b0190a339f67" />
<img width="731" height="475" alt="image" src="https://github.com/user-attachments/assets/2de25786-0b0d-4a21-abeb-c6a8c7761196" />






