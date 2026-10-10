from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import time

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset

from attention_training import CheckpointPause, atomic_save, copy_state, fingerprint, state_fingerprint
from prefix_data import FRACTIONS


MASKS = {"both": (1, 1), "video": (1, 0), "imu": (0, 1)}


def conditions():
    result = []
    for fraction in FRACTIONS:
        for modality in MASKS:
            result.append({"fraction": fraction, "modality": modality, "noise": 0.0})
        for noise in (0.1, 0.3):
            result.append({"fraction": fraction, "modality": "both", "noise": noise})
    return result


def condition_key(condition):
    return f"f{int(condition['fraction'] * 100)}-{condition['modality']}-n{int(condition['noise'] * 100)}"


def check_stop(stop):
    if stop is not None and datetime.now(timezone.utc) >= stop:
        raise CheckpointPause("Work boundary reached; saved epochs and completed stages can resume")


@dataclass(frozen=True)
class RobustConfig:
    epochs: int = 40
    batch_size: int = 16
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    dropout: float = 0.1

    def validate(self):
        if self.epochs < 1 or self.batch_size < 1 or not 0 <= self.dropout < 1:
            raise ValueError("Invalid training configuration")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        if not np.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("Invalid weight decay")


class PrefixDataset(Dataset):
    def __init__(self, video, imu, targets, indices, normalization, condition=None, variant=None, seed=42):
        self.video, self.imu, self.targets = video, imu, targets
        self.indices, self.normalization = np.asarray(indices), normalization
        self.condition = condition or {"fraction": 1.0, "modality": "both", "noise": 0.0}
        self.variant, self.seed, self.epoch = variant, seed, 0
        if variant not in (None, "full_only", "augmented"):
            raise ValueError("Unknown training variant")
        if self.condition not in conditions():
            raise ValueError("Unsupported evaluation condition")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        i = int(self.indices[index])
        condition = self.condition
        if self.variant == "augmented":
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, i]))
            condition = {"fraction": FRACTIONS[int(rng.integers(4))],
                         "modality": rng.choice(["both", "video", "imu"], p=[0.5, 0.25, 0.25]), "noise": 0.05}
        else:
            rng = np.random.default_rng(np.random.SeedSequence([2026, i, int(condition["fraction"] * 100), int(condition["noise"] * 100)]))
        j = FRACTIONS.index(condition["fraction"])
        video = np.array(self.video[i, j], dtype=np.float32, copy=True) / 255
        imu = (np.array(self.imu[i, j], dtype=np.float32, copy=True) - self.normalization["mean"][:, None]) / self.normalization["std"][:, None]
        mask = np.array(MASKS[condition["modality"]], dtype=np.float32)
        if condition["noise"] and mask[1]:
            imu += rng.normal(0, condition["noise"], imu.shape).astype(np.float32)
        if not mask[0]:
            video.fill(0)
        if not mask[1]:
            imu.fill(0)
        return torch.from_numpy(video), torch.from_numpy(imu), torch.from_numpy(mask), int(self.targets[i])


def loader(dataset, config, seed=0, shuffle=False):
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle, num_workers=0,
                      generator=torch.Generator().manual_seed(seed))


def predict_logits(model, dataset, config, device="cpu", stop=None):
    model.eval()
    output = []
    with torch.inference_mode():
        for video, imu, mask, _ in loader(dataset, config):
            check_stop(stop)
            output.append(model(video.to(device), imu.to(device), mask.to(device)).cpu().numpy())
    return np.concatenate(output)


def fit_phase(model, training, validation, config, seed, epochs, path, identity, device="cpu", stop=None,
              stop_after_epoch=None):
    config.validate()
    signature = fingerprint({"identity": identity, "config": asdict(config), "seed": seed, "epochs": epochs,
                             "device": device, "initial_state": state_fingerprint(model.state_dict()),
                             "torch": str(torch.__version__)})
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    state = {"epoch": 0, "best_epoch": epochs, "best_score": -1.0, "history": [], "seconds": 0.0}
    path = Path(path)
    if path.exists():
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state["signature"] != signature:
            raise ValueError("Training checkpoint identity changed")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["rng_cpu"])
        if device == "cuda":
            torch.cuda.set_rng_state_all(state["rng_cuda"])
    for epoch in range(state["epoch"] + 1, epochs + 1):
        check_stop(stop)
        start = time.perf_counter()
        model.train()
        training.epoch = epoch
        total, count = 0.0, 0
        for video, imu, mask, target in loader(training, config, seed + epoch, True):
            check_stop(stop)
            optimizer.zero_grad(set_to_none=True)
            logits = model(video.to(device), imu.to(device), mask.to(device))
            loss = torch.nn.functional.cross_entropy(logits, target.to(device))
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            optimizer.step()
            total += loss.item() * len(target)
            count += len(target)
        row = {"epoch": epoch, "loss": total / count}
        if validation:
            values = []
            for dataset in validation:
                logits = predict_logits(model, dataset, config, device, stop)
                values.append(float(f1_score(dataset.targets[dataset.indices], logits.argmax(1),
                                            labels=list(range(logits.shape[1])), average="macro", zero_division=0)))
            score = float(np.mean(values))
            row.update(validation_macro_f1=score, validation_condition_macro_f1=values)
            if score > state["best_score"]:
                state.update(best_score=score, best_epoch=epoch)
        state["history"].append(row)
        state.update(epoch=epoch, signature=signature, model=copy_state(model), optimizer=optimizer.state_dict(),
                     rng_cpu=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state_all() if device == "cuda" else [],
                     seconds=state["seconds"] + time.perf_counter() - start)
        atomic_save(state, path)
        print(f"{path.parent.name}/{path.stem}: epoch {epoch}/{epochs}, loss {row['loss']:.4f}", flush=True)
        if stop_after_epoch is not None and epoch >= stop_after_epoch and epoch < epochs:
            raise CheckpointPause("Paused after a saved epoch")
    return {key: state[key] for key in ("best_epoch", "best_score", "history", "seconds")}
