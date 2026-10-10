import math

import torch
from torch import nn
from torch.nn import functional as F


VARIANTS = ("concatenation", "attention", "contrastive_attention")


class CrossAttention(nn.Module):
    def __init__(self, dimension=64, heads=4, dropout=0.1):
        super().__init__()
        if dimension % heads:
            raise ValueError("Attention dimension must be divisible by head count")
        self.heads = heads
        self.query = nn.Linear(dimension, dimension)
        self.key = nn.Linear(dimension, dimension)
        self.value = nn.Linear(dimension, dimension)
        self.output = nn.Linear(dimension, dimension)
        self.norm = nn.LayerNorm(dimension)
        self.feedforward = nn.Sequential(nn.Linear(dimension, dimension * 2), nn.GELU(),
                                         nn.Dropout(dropout), nn.Linear(dimension * 2, dimension))
        self.final_norm = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, context):
        batch, count, dimension = query.shape
        def split(tensor):
            return tensor.reshape(batch, -1, self.heads, dimension // self.heads).transpose(1, 2)
        q, k, v = split(self.query(query)), split(self.key(context)), split(self.value(context))
        weights = (q @ k.transpose(-1, -2) / math.sqrt(dimension // self.heads)).softmax(dim=-1)
        attended = (self.dropout(weights) @ v).transpose(1, 2).reshape(batch, count, dimension)
        mixed = self.norm(query + self.dropout(self.output(attended)))
        return self.final_norm(mixed + self.dropout(self.feedforward(mixed))), weights


class PairedEncoder(nn.Module):
    def __init__(self, classes, variant="attention", dropout=0.1):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError("Unknown paired model variant")
        self.variant = variant
        self.video = nn.Sequential(
            nn.Conv3d(3, 16, (3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.GroupNorm(4, 16), nn.GELU(),
            nn.Conv3d(16, 32, 3, stride=(2, 4, 4), padding=1), nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv3d(32, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.GELU())
        self.imu = nn.Sequential(
            nn.Conv1d(6, 32, 7, stride=2, padding=3), nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv1d(64, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.GELU())
        self.video_position = nn.Parameter(torch.randn(1, 16, 64) * 0.02)
        self.imu_position = nn.Parameter(torch.randn(1, 16, 64) * 0.02)
        self.head = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 128), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(128, classes))
        self.video_projection = nn.Sequential(nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 32))
        self.imu_projection = nn.Sequential(nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 32))
        if variant != "concatenation":
            self.video_attention = CrossAttention(dropout=dropout)
            self.imu_attention = CrossAttention(dropout=dropout)

    def encode(self, video, imu):
        features = self.video(video)
        batch, channels, frames, height, width = features.shape
        if frames != 4 or height != 6 or width != 6:
            raise ValueError("Expected video input with 16 frames at 96 by 96")
        video_tokens = features.reshape(batch, channels, frames, 2, 3, 2, 3).mean(dim=(4, 6))
        video_tokens = video_tokens.flatten(2).transpose(1, 2)
        imu_tokens = self.imu(imu).transpose(1, 2)
        if imu_tokens.shape[1] != 16:
            raise ValueError("Expected 128 IMU samples")
        return video_tokens, imu_tokens

    def contrastive_embeddings(self, video, imu):
        video_tokens, imu_tokens = self.encode(video, imu)
        return (F.normalize(self.video_projection(video_tokens.mean(dim=1)), dim=-1),
                F.normalize(self.imu_projection(imu_tokens.mean(dim=1)), dim=-1))

    def forward(self, video, imu, return_attention=False):
        video_tokens, imu_tokens = self.encode(video, imu)
        video_tokens, imu_tokens = video_tokens + self.video_position, imu_tokens + self.imu_position
        evidence = {}
        if self.variant != "concatenation":
            attended_video, evidence["video_to_imu"] = self.video_attention(video_tokens, imu_tokens)
            attended_imu, evidence["imu_to_video"] = self.imu_attention(imu_tokens, video_tokens)
            video_tokens, imu_tokens = attended_video, attended_imu
        logits = self.head(torch.cat((video_tokens.mean(dim=1), imu_tokens.mean(dim=1)), dim=1))
        return (logits, evidence) if return_attention else logits


def paired_contrastive_loss(video, imu, temperature=0.1):
    if video.ndim != 2 or video.shape != imu.shape or len(video) < 2:
        raise ValueError("Contrastive loss requires at least two matching embedding pairs")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    video, imu = F.normalize(video, dim=-1), F.normalize(imu, dim=-1)
    similarities = video @ imu.T / temperature
    targets = torch.arange(len(video), device=video.device)
    return (F.cross_entropy(similarities, targets) + F.cross_entropy(similarities.T, targets)) / 2
