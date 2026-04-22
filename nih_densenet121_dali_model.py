import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import glob
import time
from dataclasses import dataclass
from multiprocessing import freeze_support
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torchvision.models import DenseNet121_Weights, densenet121

from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    hamming_loss,
)

import matplotlib.pyplot as plt


# ______________________________________________________________________________________
# NIH metadata + labels

class NIHDatasetMetadata:
    def __init__(self, root_dir: str, csv_file: str = "Data_Entry_2017.csv"):
        self.root_dir = root_dir

        csv_path = os.path.join(root_dir, csv_file)
        self.data = pd.read_csv(csv_path)
        self.data["Finding Labels"] = self.data["Finding Labels"].fillna("")

        self.all_labels = sorted(
            set(self.data["Finding Labels"].str.split("|").explode().str.strip()) - {""}
        )
        self.class_to_idx = {label: idx for idx, label in enumerate(self.all_labels)}
        self.idx_to_class = {idx: label for label, idx in self.class_to_idx.items()}

        self.image_paths = {}
        image_folders = sorted(glob.glob(os.path.join(root_dir, "images_*")))
        for folder in image_folders:
            pngs = glob.glob(os.path.join(folder, "**", "*.png"), recursive=True)
            for img_path in pngs:
                self.image_paths[os.path.basename(img_path)] = img_path

        self.data = self.data[self.data["Image Index"].isin(self.image_paths.keys())].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.data)


@dataclass
class Sample:
    image_path: str
    target: np.ndarray


def build_samples(dataset: NIHDatasetMetadata) -> List[Sample]:
    samples: List[Sample] = []

    for _, row in dataset.data.iterrows():
        image_name = row["Image Index"]
        image_path = dataset.image_paths[image_name]

        target = np.zeros(len(dataset.all_labels), dtype=np.float32)
        labels = row["Finding Labels"]
        if labels:
            for label in labels.split("|"):
                label = label.strip()
                if label in dataset.class_to_idx:
                    target[dataset.class_to_idx[label]] = 1.0

        samples.append(Sample(image_path=image_path, target=target))

    return samples


# ______________________________________________________________________________________
# Split helpers

def split_samples(samples: Sequence[Sample], val_fraction: float, test_fraction: float, seed: int = 42):
    total_size = len(samples)
    val_size = int(total_size * val_fraction)
    test_size = int(total_size * test_fraction)
    train_size = total_size - val_size - test_size

    if train_size <= 0:
        raise ValueError("Train split became empty. Reduce val_fraction and/or test_fraction.")

    rng = np.random.default_rng(seed)
    indices = np.arange(total_size)
    rng.shuffle(indices)

    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]

    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]
    test_samples = [samples[i] for i in test_idx]
    return train_samples, val_samples, test_samples


# ______________________________________________________________________________________
# Class imbalance weighting

def compute_pos_weight_from_samples(samples: Sequence[Sample]) -> torch.Tensor:
    targets = np.stack([sample.target for sample in samples], axis=0)
    class_counts = torch.tensor(targets.sum(axis=0), dtype=torch.float32)
    total_samples = targets.shape[0]
    neg_counts = total_samples - class_counts
    class_counts = torch.clamp(class_counts, min=1.0)
    return neg_counts / class_counts


# ______________________________________________________________________________________
# Model

class NIHDenseNet121(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        weights = DenseNet121_Weights.DEFAULT if pretrained else None
        self.model = densenet121(weights=weights)
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)


def initialize_model(num_classes: int, pos_weight: torch.Tensor, lr: float, pretrained: bool):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NIHDenseNet121(num_classes=num_classes, pretrained=pretrained).to(device)
    loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = optim.Adam(model.parameters(), lr=lr)
    return model, loss_function, optimizer, device


# ______________________________________________________________________________________
# DALI external source

class NIHExternalInputIterator:
    def __init__(self, samples: Sequence[Sample], batch_size: int, shuffle: bool):
        self.samples = list(samples)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(self.samples)
        self.indices = np.arange(self.num_samples)
        self.position = 0
        self.reset()

    def reset(self):
        self.position = 0
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __iter__(self):
        self.reset()
        return self

    def __next__(self):
        if self.position >= self.num_samples:
            raise StopIteration

        batch_indices = self.indices[self.position:self.position + self.batch_size]
        self.position += self.batch_size

        encoded_images = []
        labels = []

        for idx in batch_indices:
            sample = self.samples[idx]
            with open(sample.image_path, "rb") as f:
                encoded_images.append(np.frombuffer(f.read(), dtype=np.uint8))
            labels.append(sample.target.astype(np.float32, copy=False))

        labels = np.ascontiguousarray(np.stack(labels, axis=0), dtype=np.float32)
        return encoded_images, labels


@pipeline_def
def nih_dali_pipeline(external_data, training: bool, mean: Sequence[float], std: Sequence[float]):
    encoded_images, labels = fn.external_source(
        source=external_data,
        num_outputs=2,
        batch=True,
        parallel=False,
        dtype=[types.UINT8, types.FLOAT],
    )

    images = fn.decoders.image(encoded_images, device="cpu", output_type=types.RGB)
    images = images.gpu()
    images = fn.resize(images, device="gpu", resize_x=224, resize_y=224)

    if training:
        angle = fn.random.uniform(range=(-1.0, 1.0))
        images = fn.rotate(images, device="gpu", angle=angle, fill_value=0, keep_size=True)

    images = fn.crop_mirror_normalize(
        images,
        device="gpu",
        dtype=types.FLOAT,
        output_layout="CHW",
        mean=[m * 255.0 for m in mean],
        std=[s * 255.0 for s in std],
    )

    labels = labels.gpu()
    return images, labels


class DALILoaderWrapper:
    def __init__(self, samples: Sequence[Sample], batch_size: int, training: bool, num_threads: int, device_id: int,
                 mean: Sequence[float], std: Sequence[float]):
        self.samples = list(samples)
        self.batch_size = batch_size
        self.training = training
        self.num_threads = num_threads
        self.device_id = device_id
        self.mean = mean
        self.std = std
        self._build()

    def _build(self):
        self.external_data = NIHExternalInputIterator(
            samples=self.samples,
            batch_size=self.batch_size,
            shuffle=self.training,
        )
        self.pipe = nih_dali_pipeline(
            external_data=self.external_data,
            training=self.training,
            mean=self.mean,
            std=self.std,
            batch_size=self.batch_size,
            num_threads=self.num_threads,
            device_id=self.device_id,
            prefetch_queue_depth=2,
        )

        

        self.pipe.build()
        self.iterator = DALIGenericIterator(
            pipelines=[self.pipe],
            output_map=["inputs", "targets"],
            size=-1,
            auto_reset=True,
            prepare_first_batch=False,
        )

    def __iter__(self):
        self.reset()
        return iter(self.iterator)

    def __len__(self):
        return (len(self.samples) + self.batch_size - 1) // self.batch_size

    def reset(self):
        self.iterator.reset()
        self.external_data.reset()


# ______________________________________________________________________________________
# Train / eval

def train_epoch(model, loss_function, optimizer, train_loader, device):
    model.train()
    epoch_loss = 0.0
    running_loss = 0.0
    num_correct_pred = 0
    total_pred = 0
    total_batches = len(train_loader)
    epoch_start_time = time.time()
    divider = 96

    for batch_idx, data in enumerate(train_loader):
        batch = data[0]
        inputs = batch["inputs"]
        targets = batch["targets"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        epoch_loss += loss_value
        running_loss += loss_value

        probs = torch.sigmoid(outputs)
        predicted_values = (probs > 0.5).float()
        batch_elements = targets.numel()
        total_pred += batch_elements
        num_correct_pred += (predicted_values == targets).sum().item()

        elapsed_epoch_time = time.time() - epoch_start_time

        if (batch_idx + 1) % divider == 0 or (batch_idx + 1) == total_batches:
            batches_in_window = divider if (batch_idx + 1) % divider == 0 else (batch_idx + 1) % divider
            avg_running_loss = running_loss / batches_in_window
            accuracy = (num_correct_pred / max(total_pred, 1)) * 100
            avg_time_per_batch = elapsed_epoch_time / (batch_idx + 1)
            print(
                f"\tStep {batch_idx + 1}/{total_batches} - Loss: {avg_running_loss:.3f} | "
                f"Acc: {accuracy:.2f}% | Time Elapsed: {elapsed_epoch_time/60:.2f}Mins | "
                f"Avg Time Per Batch: {avg_time_per_batch:.2f}Secs"
            )
            running_loss = 0.0
            num_correct_pred = 0
            total_pred = 0

    train_loader.reset()
    return epoch_loss / max(total_batches, 1)


@torch.no_grad()
def evaluate(model, data_loader, loss_function, device, class_names, split_name: str = "Validation", threshold: float = 0.5):
    model.eval()

    all_targets = []
    all_probs = []
    total_loss = 0.0
    total_batches = 0

    for data in data_loader:
        batch = data[0]
        inputs = batch["inputs"]
        targets = batch["targets"].to(device, non_blocking=True)

        outputs = model(inputs)
        loss = loss_function(outputs, targets)

        probs = torch.sigmoid(outputs)

        total_loss += loss.item()
        total_batches += 1

        all_targets.append(targets.detach().cpu())
        all_probs.append(probs.detach().cpu())

    data_loader.reset()

    y_true = torch.cat(all_targets, dim=0).numpy()
    y_prob = torch.cat(all_probs, dim=0).numpy()
    y_pred = (y_prob >= threshold).astype(np.float32)

    avg_loss = total_loss / max(total_batches, 1)

    # label-wise accuracy kept only as a supporting metric
    label_wise_accuracy = (y_pred == y_true).mean() * 100.0

    per_class_auroc = {}
    per_class_auprc = {}

    valid_aurocs = []
    valid_auprcs = []

    for class_idx, class_name in enumerate(class_names):
        y_true_c = y_true[:, class_idx]
        y_prob_c = y_prob[:, class_idx]

        # AUROC / AUPRC need both positive and negative samples for that class
        if len(np.unique(y_true_c)) < 2:
            per_class_auroc[class_name] = None
            per_class_auprc[class_name] = None
            continue

        auroc = roc_auc_score(y_true_c, y_prob_c)
        auprc = average_precision_score(y_true_c, y_prob_c)

        per_class_auroc[class_name] = float(auroc)
        per_class_auprc[class_name] = float(auprc)

        valid_aurocs.append(auroc)
        valid_auprcs.append(auprc)

    macro_auroc = float(np.mean(valid_aurocs)) if valid_aurocs else None
    macro_auprc = float(np.mean(valid_auprcs)) if valid_auprcs else None

    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    micro_precision = precision_score(y_true, y_pred, average="micro", zero_division=0)
    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)

    micro_recall = recall_score(y_true, y_pred, average="micro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)

    ham_loss = hamming_loss(y_true, y_pred)

    metrics = {
        "loss": float(avg_loss),
        "label_wise_accuracy": float(label_wise_accuracy),
        "macro_auroc": macro_auroc,
        "macro_auprc": macro_auprc,
        "micro_f1": float(micro_f1),
        "macro_f1": float(macro_f1),
        "micro_precision": float(micro_precision),
        "macro_precision": float(macro_precision),
        "micro_recall": float(micro_recall),
        "macro_recall": float(macro_recall),
        "hamming_loss": float(ham_loss),
        "per_class_auroc": per_class_auroc,
        "per_class_auprc": per_class_auprc,
    }

    print(f"\n\t[{split_name}]")
    print(f"\tLoss: {metrics['loss']:.4f}")
    print(f"\tLabel-wise Accuracy: {metrics['label_wise_accuracy']:.2f}%")
    print(f"\tMacro AUROC: {metrics['macro_auroc']:.4f}" if metrics["macro_auroc"] is not None else "\tMacro AUROC: N/A")
    print(f"\tMacro AUPRC: {metrics['macro_auprc']:.4f}" if metrics["macro_auprc"] is not None else "\tMacro AUPRC: N/A")
    print(f"\tMicro F1: {metrics['micro_f1']:.4f}")
    print(f"\tMacro F1: {metrics['macro_f1']:.4f}")
    print(f"\tMicro Precision: {metrics['micro_precision']:.4f}")
    print(f"\tMacro Precision: {metrics['macro_precision']:.4f}")
    print(f"\tMicro Recall: {metrics['micro_recall']:.4f}")
    print(f"\tMacro Recall: {metrics['macro_recall']:.4f}")
    print(f"\tHamming Loss: {metrics['hamming_loss']:.4f}")

    print(f"\n\tPer-class AUROC ({split_name}):")
    for class_name, value in metrics["per_class_auroc"].items():
        if value is None:
            print(f"\t  {class_name}: N/A")
        else:
            print(f"\t  {class_name}: {value:.4f}")

    print(f"\n\tPer-class AUPRC ({split_name}):")
    for class_name, value in metrics["per_class_auprc"].items():
        if value is None:
            print(f"\t  {class_name}: N/A")
        else:
            print(f"\t  {class_name}: {value:.4f}")

    return metrics



#______________________________________________________________________________

def plot_training_history(history):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], marker="o", label="Train Loss")
    plt.plot(epochs, history["val_loss"], marker="o", label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_curve.png", dpi=300, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["val_macro_auroc"], marker="o", label="Validation Macro AUROC")
    plt.xlabel("Epoch")
    plt.ylabel("Macro AUROC")
    plt.title("Validation Macro AUROC")
    plt.legend()
    plt.grid(True)
    plt.savefig("val_macro_auroc.png", dpi=300, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["val_macro_auprc"], marker="o", label="Validation Macro AUPRC")
    plt.xlabel("Epoch")
    plt.ylabel("Macro AUPRC")
    plt.title("Validation Macro AUPRC")
    plt.legend()
    plt.grid(True)
    plt.savefig("val_macro_auprc.png", dpi=300, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["val_micro_f1"], marker="o", label="Validation Micro F1")
    plt.plot(epochs, history["val_macro_f1"], marker="o", label="Validation Macro F1")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.title("Validation F1 Scores")
    plt.legend()
    plt.grid(True)
    plt.savefig("val_f1_scores.png", dpi=300, bbox_inches="tight")
    plt.show()

# ______________________________________________________________________________________
# Args / main

def parse_args():
    parser = argparse.ArgumentParser(description="NIH Chest X-ray DenseNet121 trainer with NVIDIA DALI")
    parser.add_argument("--data-root", type=str, required=True, help="Path to NIH_Xray folder")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dali-threads", type=int, default=4, help="Number of DALI worker threads")
    parser.add_argument("--pretrained", action="store_true", default=True)
    return parser.parse_args()


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this DALI training script.")

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    print(f"Using data root: {data_root}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Torch CPU threads: {torch.get_num_threads()}")
    print(f"DALI threads: {args.dali_threads}")

    mean = [0.4981, 0.4981, 0.4981]
    std = [0.2482, 0.2482, 0.2482]

    dataset = NIHDatasetMetadata(root_dir=str(data_root), csv_file="Data_Entry_2017.csv")
    print(f"Dataset size: {len(dataset)}")
    print(f"Num classes: {len(dataset.all_labels)}")

    samples = build_samples(dataset)
    train_samples, val_samples, test_samples = split_samples(
        samples=samples,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=42,
    )

    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples: {len(val_samples)}")
    print(f"Test samples: {len(test_samples)}")

    device_id = torch.cuda.current_device()

    train_loader = DALILoaderWrapper(
        samples=train_samples,
        batch_size=args.batch_size,
        training=True,
        num_threads=args.dali_threads,
        device_id=device_id,
        mean=mean,
        std=std,
    )
    val_loader = DALILoaderWrapper(
        samples=val_samples,
        batch_size=args.batch_size,
        training=False,
        num_threads=args.dali_threads,
        device_id=device_id,
        mean=mean,
        std=std,
    )
    test_loader = DALILoaderWrapper(
        samples=test_samples,
        batch_size=args.batch_size,
        training=False,
        num_threads=args.dali_threads,
        device_id=device_id,
        mean=mean,
        std=std,
    )

    pos_weight = compute_pos_weight_from_samples(train_samples)
    model, loss_function, optimizer, device = initialize_model(
        num_classes=len(dataset.all_labels),
        pos_weight=pos_weight,
        lr=args.lr,
        pretrained=args.pretrained,
    )

    history = {
    "train_loss": [],
    "val_loss": [],
    "val_macro_auroc": [],
    "val_macro_auprc": [],
    "val_micro_f1": [],
    "val_macro_f1": [],
    "val_micro_precision": [],
    "val_macro_precision": [],
    "val_micro_recall": [],
    "val_macro_recall": [],
    "val_hamming_loss": [],
}


    
    train_losses = []
    val_history = []
    best_val_macro_auroc = -1.0
    best_model_path = "best_nih_densenet121.pth"

    for epoch in range(args.epochs):
        print(f"\n================ Epoch {epoch + 1}/{args.epochs} ================")

        print(f"[Training] Epoch {epoch + 1}:")
        loss = train_epoch(model, loss_function, optimizer, train_loader, device)
        train_losses.append(loss)

        print(f"[Validation] Epoch {epoch + 1}:")
        val_metrics = evaluate(
            model=model,
            data_loader=val_loader,
            loss_function=loss_function,
            device=device,
            class_names=dataset.all_labels,
            split_name="Validation",
            threshold=0.5,
        )
        val_history.append(val_metrics)
        history["train_loss"].append(loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_macro_auroc"].append(val_metrics["macro_auroc"])
        history["val_macro_auprc"].append(val_metrics["macro_auprc"])
        history["val_micro_f1"].append(val_metrics["micro_f1"])
        history["val_macro_f1"].append(val_metrics["macro_f1"])
        history["val_micro_precision"].append(val_metrics["micro_precision"])
        history["val_macro_precision"].append(val_metrics["macro_precision"])
        history["val_micro_recall"].append(val_metrics["micro_recall"])
        history["val_macro_recall"].append(val_metrics["macro_recall"])
        history["val_hamming_loss"].append(val_metrics["hamming_loss"])

        current_val_macro_auroc = val_metrics["macro_auroc"]
        if current_val_macro_auroc is not None and current_val_macro_auroc > best_val_macro_auroc:
            best_val_macro_auroc = current_val_macro_auroc

            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "num_classes": len(dataset.all_labels),
                "class_to_idx": dataset.class_to_idx,
                "idx_to_class": dataset.idx_to_class,
                "mean": mean,
                "std": std,
                "image_size": (224, 224),
                "best_val_macro_auroc": best_val_macro_auroc,
            }, best_model_path)

            print(f"\nSaved best model to: {best_model_path}")
            print(f"Best Validation Macro AUROC so far: {best_val_macro_auroc:.4f}")

    print("\n================ Final summary ================")
    print(f"Train losses: {[round(x, 4) for x in train_losses]}")
    print(f"Best Validation Macro AUROC: {best_val_macro_auroc:.4f}" if best_val_macro_auroc >= 0 else "Best Validation Macro AUROC: N/A")

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print("\n[Test evaluation using best saved model]")
    test_metrics = evaluate(
        model=model,
        data_loader=test_loader,
        loss_function=loss_function,
        device=device,
        class_names=dataset.all_labels,
        split_name="Test",
        threshold=0.5,
    )

    plot_training_history(history)








if __name__ == "__main__":
    freeze_support()
    main()
