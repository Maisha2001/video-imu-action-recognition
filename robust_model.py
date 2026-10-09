import torch
from torch import nn

from attention_model import PairedEncoder


class RobustEncoder(PairedEncoder):
    def __init__(self, classes=21, dropout=0.1):
        super().__init__(classes, "attention", dropout)
        self.head = nn.Sequential(nn.LayerNorm(130), nn.Linear(130, 128), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(128, classes))

    def forward(self, video, imu, available):
        if available.shape != (len(video), 2):
            raise ValueError("Expected one video/IMU availability mask per example")
        if not torch.all((available == 0) | (available == 1)) or not torch.all(available.sum(1) > 0):
            raise ValueError("At least one modality must be available; mask values must be binary")
        vm, sm = available[:, 0, None, None], available[:, 1, None, None]
        video = torch.where(vm[:, :, :, None, None].bool(), video, 0)
        imu = torch.where(sm.bool(), imu, 0)
        vt, st = self.encode(video, imu)
        vt, st = (vt + self.video_position) * vm, (st + self.imu_position) * sm
        av, _ = self.video_attention(vt, st)
        ai, _ = self.imu_attention(st, vt)
        vt = (av * sm + vt * (1 - sm)) * vm
        st = (ai * vm + st * (1 - vm)) * sm
        return self.head(torch.cat((vt.mean(1), st.mean(1), available), dim=1))
