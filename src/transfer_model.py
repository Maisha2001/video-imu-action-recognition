import numpy as np
import torch
from torch import nn

from prefix_data import FRACTIONS
from robust_training import MASKS, conditions


class TransferFusion(nn.Module):
    def __init__(self, classes=21, dropout=0.1):
        super().__init__()
        self.video = nn.Sequential(nn.Linear(512, 128), nn.LayerNorm(128), nn.GELU())
        self.imu = nn.Sequential(nn.Conv1d(6, 32, 7, stride=2, padding=3), nn.GroupNorm(8, 32), nn.GELU(),
                                 nn.Conv1d(32, 64, 5, stride=2, padding=2), nn.GroupNorm(8, 64), nn.GELU(),
                                 nn.Conv1d(64, 128, 3, stride=2, padding=1), nn.GroupNorm(8, 128), nn.GELU())
        self.head = nn.Sequential(nn.LayerNorm(258), nn.Linear(258, 128), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(128, classes))

    def forward(self, video, imu, available):
        if available.shape != (len(video), 2) or not torch.all((available == 0) | (available == 1)):
            raise ValueError("Expected binary modality masks")
        if not torch.all(available.sum(1) > 0):
            raise ValueError("At least one sensor must be available")
        vm, sm = available[:, :1], available[:, 1:]
        video = torch.where(vm.bool(), video, 0)
        imu = torch.where(sm[:, :, None].bool(), imu, 0)
        visual = self.video(video) * vm
        sensor = self.imu(imu).mean(-1) * sm
        return self.head(torch.cat((visual, sensor, available), dim=1))


def fit_feature_normalization(features, indices):
    selected = np.asarray(features[indices], dtype=np.float64)
    if not len(selected) or not np.isfinite(selected).all():
        raise ValueError("Feature normalization needs finite training examples")
    return {"mean": selected.mean((0, 1)).astype(np.float32),
            "std": np.maximum(selected.std((0, 1)), 1e-6).astype(np.float32)}


class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, features, imu, targets, indices, normalization, condition=None, augment=False, seed=42):
        self.features, self.imu, self.targets = features, imu, targets
        self.indices, self.normalization = np.asarray(indices), normalization
        self.condition = condition or {"fraction": 1.0, "modality": "both", "noise": 0.0}
        self.augment, self.seed, self.epoch = augment, seed, 0
        if self.condition not in conditions():
            raise ValueError("Unsupported condition")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        i, condition = int(self.indices[index]), self.condition
        if self.augment:
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, i]))
            condition = {"fraction": FRACTIONS[int(rng.integers(4))],
                         "modality": rng.choice(["both", "video", "imu"], p=[0.5, 0.25, 0.25]), "noise": 0.05}
        else:
            rng = np.random.default_rng(np.random.SeedSequence([2026, i, int(condition["fraction"] * 100), int(condition["noise"] * 100)]))
        j = FRACTIONS.index(condition["fraction"])
        norm = self.normalization
        visual = (self.features[i, j].copy() - norm["video"]["mean"]) / norm["video"]["std"]
        sensor = (self.imu[i, j].copy() - norm["imu"]["mean"][:, None]) / norm["imu"]["std"][:, None]
        mask = np.array(MASKS[condition["modality"]], dtype=np.float32)
        if condition["noise"] and mask[1]:
            sensor += rng.normal(0, condition["noise"], sensor.shape).astype(np.float32)
        if not mask[0]:
            visual.fill(0)
        if not mask[1]:
            sensor.fill(0)
        return torch.from_numpy(visual), torch.from_numpy(sensor), torch.from_numpy(mask), int(self.targets[i])
