from dataclasses import asdict, dataclass
import os
import random

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 30
    batch_size: int = 16
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    seed: int = 42

    def validate(self):
        if self.epochs < 1 or self.batch_size < 2 or not 0 <= self.seed < 2 ** 32:
            raise ValueError("Invalid epochs, batch size, or seed")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("Learning rate must be finite and positive")
        if not np.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("Weight decay must be finite and nonnegative")


def seed_training(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


class GlobalMean(nn.Module):
    def forward(self, x):
        return x.mean(dim=tuple(range(2, x.ndim)))


class VideoEncoder(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(3, 16, (3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.BatchNorm3d(16), nn.ReLU(),
            nn.Conv3d(16, 32, 3, stride=(2, 4, 4), padding=1), nn.BatchNorm3d(32), nn.ReLU(),
            nn.Conv3d(32, 64, 3, stride=2, padding=1), nn.BatchNorm3d(64), nn.ReLU(),
            GlobalMean())
        self.head = nn.Linear(64, classes)

    def forward(self, x):
        return self.head(self.encoder(x))


class ImuEncoder(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(6, 32, 7, stride=2, padding=3), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 3, stride=2, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            GlobalMean())
        self.head = nn.Linear(128, classes)

    def forward(self, x):
        return self.head(self.encoder(x))


def make_model(modality, classes):
    if modality not in {"video", "imu"}:
        raise ValueError("Unknown modality")
    return (VideoEncoder if modality == "video" else ImuEncoder)(classes)


def fit_normalization(imu, indices):
    selected = np.asarray(imu[indices], dtype=np.float64)
    if not len(selected) or not np.isfinite(selected).all():
        raise ValueError("Normalization needs finite training data")
    return {"mean": selected.mean(axis=(0, 2)).astype(np.float32),
            "std": np.maximum(selected.std(axis=(0, 2)), 1e-6).astype(np.float32)}


class TrialDataset(Dataset):
    def __init__(self, values, y, indices, modality, normalization):
        self.values, self.y, self.indices = values, y, indices
        self.modality, self.normalization = modality, normalization

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        i = self.indices[index]
        value = np.array(self.values[i], dtype=np.float32, copy=True)
        if self.modality == "video":
            value /= 255.0
        else:
            value = (value - self.normalization["mean"][:, None]) / self.normalization["std"][:, None]
        return torch.from_numpy(value), int(self.y[i])


def loader(values, y, indices, modality, normalization, config, shuffle=False):
    return DataLoader(TrialDataset(values, y, indices, modality, normalization),
                      batch_size=config.batch_size, shuffle=shuffle, num_workers=0,
                      generator=torch.Generator().manual_seed(config.seed))


def probabilities(model, batches, device):
    model.eval()
    outputs = []
    with torch.inference_mode():
        for x, _ in batches:
            outputs.append(model(x.to(device)).softmax(dim=1).cpu().numpy())
    return np.concatenate(outputs)


def train_encoder(values, y, train, validation, modality, normalization, classes, config, device):
    config.validate()
    seed_training(config.seed)
    model = make_model(modality, classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    batches = loader(values, y, train, modality, normalization, config, shuffle=True)
    validation_batches = loader(values, y, validation, modality, normalization, config) if len(validation) else None
    history, best_score, best_epoch, best_state, best_prob = [], -1.0, config.epochs, None, None
    for epoch in range(1, config.epochs + 1):
        model.train()
        loss_sum = 0.0
        for x, target in batches:
            x, target = x.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(x), target)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(target)
        row = {"epoch": epoch, "train_loss": loss_sum / len(train)}
        if validation_batches is not None:
            prob = probabilities(model, validation_batches, device)
            score = float(f1_score(y[validation], prob.argmax(axis=1), labels=list(range(classes)),
                                   average="macro", zero_division=0))
            row["validation_macro_f1"] = score
            if score > best_score:
                best_score, best_epoch, best_prob = score, epoch, prob
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        history.append(row)
        print(f"{modality} epoch {epoch}/{config.epochs}: {row}", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"selected_epoch": best_epoch, "history": history,
                   "validation_probabilities": best_prob.tolist() if best_prob is not None else None,
                   "parameters": sum(p.numel() for p in model.parameters()), "config": asdict(config)}


def fuse(video, imu, weight):
    video, imu = np.asarray(video), np.asarray(imu)
    if video.ndim != 2 or video.shape != imu.shape or not 0 <= weight <= 1:
        raise ValueError("Invalid fusion inputs")
    for prob in (video, imu):
        if not np.isfinite(prob).all() or (prob < 0).any() or not np.allclose(prob.sum(axis=1), 1, atol=1e-5):
            raise ValueError("Expected finite class probabilities")
    return weight * video + (1 - weight) * imu


def select_fusion(video, imu, y):
    candidates = []
    for weight in (0.0, 0.25, 0.5, 0.75, 1.0):
        prediction = fuse(video, imu, weight).argmax(axis=1)
        score = float(f1_score(y, prediction, labels=list(range(video.shape[1])), average="macro", zero_division=0))
        candidates.append({"video_weight": weight, "macro_f1": score})
    chosen = max(candidates, key=lambda row: (row["macro_f1"], -abs(row["video_weight"] - 0.5), -row["video_weight"]))
    return chosen["video_weight"], candidates
