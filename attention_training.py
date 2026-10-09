from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset

from attention_model import paired_contrastive_loss
from deep_model import seed_training


@dataclass(frozen=True)
class AttentionConfig:
    epochs: int = 40
    pretrain_epochs: int = 20
    batch_size: int = 16
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    dropout: float = 0.1
    temperature: float = 0.1

    def validate(self):
        if self.epochs < 1 or self.pretrain_epochs < 0 or self.batch_size < 2:
            raise ValueError("Invalid epoch count or batch size")
        if not 0 <= self.dropout < 1:
            raise ValueError("Dropout must be in [0, 1)")
        for value in (self.learning_rate, self.temperature):
            if not np.isfinite(value) or value <= 0:
                raise ValueError("Learning rate and temperature must be finite and positive")
        if not np.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("Invalid weight decay")


def configure(seed):
    seed_training(seed)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def fingerprint(value):
    return sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def state_fingerprint(state):
    digest = sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(path)


def copy_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


class PairedDataset(Dataset):
    def __init__(self, video, imu, y, indices, normalization):
        self.video, self.imu, self.y = video, imu, y
        self.indices, self.normalization = indices, normalization

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        i = self.indices[index]
        video = np.array(self.video[i], dtype=np.float32, copy=True) / 255
        imu = (np.array(self.imu[i], dtype=np.float32, copy=True) - self.normalization["mean"][:, None]) / self.normalization["std"][:, None]
        return torch.from_numpy(video), torch.from_numpy(imu), int(self.y[i])


def batches(dataset, config, seed=0, shuffle=False, contrastive=False):
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle, num_workers=0,
                      drop_last=contrastive and len(dataset) % config.batch_size == 1,
                      generator=torch.Generator().manual_seed(seed))


def predict_probabilities(model, dataset, config, device):
    model.eval()
    result = []
    with torch.inference_mode():
        for video, imu, _ in batches(dataset, config):
            result.append(model(video.to(device), imu.to(device)).softmax(dim=1).cpu().numpy())
    return np.concatenate(result)


class CheckpointPause(RuntimeError):
    pass


def train_phase(model, train, validation, config, seed, epochs, objective, path, identity,
                device="cpu", stop_after_epoch=None, stop_at_utc=None):
    config.validate()
    if objective not in {"classification", "contrastive"} or epochs < 1:
        raise ValueError("Invalid training phase")
    if objective == "contrastive" and len(train) < 2:
        raise ValueError("Contrastive training requires multiple pairs")
    path = Path(path)
    spec = {"identity": identity, "config": asdict(config), "seed": seed, "epochs": epochs,
            "objective": objective, "device": device, "torch": str(torch.__version__),
            "initial_state_sha256": state_fingerprint(model.state_dict())}
    signature = fingerprint(spec)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    state = {"epoch": 0, "best_score": -1.0, "best_epoch": epochs, "history": [],
             "best_model": None, "best_probabilities": None, "seconds": 0.0}
    if path.exists():
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state["signature"] != signature:
            raise ValueError("Checkpoint identity, source, device, or configuration differs")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["rng_cpu"])
        if device == "cuda":
            torch.cuda.set_rng_state_all(state["rng_cuda"])
    for epoch in range(state["epoch"] + 1, epochs + 1):
        if stop_at_utc is not None and datetime.now(timezone.utc) >= stop_at_utc:
            raise CheckpointPause("Stopped before a new epoch; resume from the saved checkpoint")
        started = time.perf_counter()
        model.train()
        loss_sum, count = 0.0, 0
        for video, imu, target in batches(train, config, seed + epoch, True, objective == "contrastive"):
            video, imu, target = video.to(device), imu.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            if objective == "contrastive":
                loss = paired_contrastive_loss(*model.contrastive_embeddings(video, imu), config.temperature)
            else:
                loss = torch.nn.functional.cross_entropy(model(video, imu), target)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(target)
            count += len(target)
        row = {"epoch": epoch, "loss": loss_sum / count}
        if validation is not None and objective == "classification":
            probability = predict_probabilities(model, validation, config, device)
            truth = np.array([validation.y[i] for i in validation.indices])
            score = float(f1_score(truth, probability.argmax(1), labels=list(range(probability.shape[1])),
                                   average="macro", zero_division=0))
            row["validation_macro_f1"] = score
            if score > state["best_score"]:
                state.update(best_score=score, best_epoch=epoch, best_model=copy_state(model),
                             best_probabilities=probability.tolist())
        state["history"].append(row)
        state.update(epoch=epoch, model=copy_state(model), optimizer=optimizer.state_dict(),
                     rng_cpu=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state_all() if device == "cuda" else [],
                     signature=signature, spec=spec, seconds=state["seconds"] + time.perf_counter() - started)
        atomic_save(state, path)
        print(f"{path.stem}: epoch {epoch}/{epochs}, loss {row['loss']:.4f}" +
              (f", validation F1 {row['validation_macro_f1']:.4f}" if "validation_macro_f1" in row else ""), flush=True)
        if stop_after_epoch is not None and epoch >= stop_after_epoch and epoch < epochs:
            raise CheckpointPause("Requested pause after saving the epoch")
    if state["best_model"] is not None:
        model.load_state_dict(state["best_model"])
    return {key: state[key] for key in ("best_epoch", "best_score", "history", "best_probabilities", "seconds")}
