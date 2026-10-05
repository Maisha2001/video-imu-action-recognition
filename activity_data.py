from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path, PurePosixPath
import re
from zipfile import ZipFile

import numpy as np
from scipy.io import loadmat


PATTERN = re.compile(r"a(\d+)_s(\d+)_t(\d+)_(inertial|color)\.(mat|avi)")
DEVELOPMENT_SUBJECTS = (1, 3, 5)
VALIDATION_SUBJECTS = (7,)
TEST_SUBJECTS = (2, 4, 6, 8)
MISSING_TRIALS = {(8, 1, 4), (23, 6, 4), (27, 8, 4)}


@dataclass(frozen=True, order=True)
class Trial:
    action: int
    subject: int
    repetition: int

    @property
    def key(self):
        return f"a{self.action}_s{self.subject}_t{self.repetition}"


def file_hash(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def archive_index(archive, modality):
    if modality not in {"inertial", "color"}:
        raise ValueError("Unsupported modality")
    index = {}
    extension = ".mat" if modality == "inertial" else ".avi"
    with ZipFile(archive) as bundle:
        for entry in bundle.infolist():
            name = PurePosixPath(entry.filename.replace("\\", "/"))
            if entry.is_dir() or name.suffix.lower() != extension:
                continue
            if name.is_absolute() or ".." in name.parts:
                raise ValueError(f"Unsafe archive member: {entry.filename}")
            match = PATTERN.fullmatch(name.name)
            if not match or match[4] != modality or "." + match[5] != extension:
                raise ValueError(f"Unexpected trial name: {name.name}")
            trial = Trial(*map(int, match.groups()[:3]))
            if not (1 <= trial.action <= 27 and 1 <= trial.subject <= 8
                    and 1 <= trial.repetition <= 4):
                raise ValueError(f"Trial outside dataset bounds: {trial.key}")
            if trial in index:
                raise ValueError(f"Duplicate trial: {trial.key}")
            if entry.file_size == 0:
                raise ValueError(f"Empty trial: {trial.key}")
            index[trial] = entry.filename
    if not index:
        raise ValueError(f"No {modality} trials found")
    return dict(sorted(index.items()))


def validate_signal(signal):
    array = np.asarray(signal, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 6 or array.shape[0] < 8:
        raise ValueError("Expected at least eight samples with six IMU channels")
    if not np.isfinite(array).all():
        raise ValueError("IMU samples must be finite")
    return array


def read_signal(bundle, member):
    entry = bundle.getinfo(member)
    if entry.file_size > 10 * 1024 * 1024:
        raise ValueError("IMU trial exceeds the 10 MiB input limit")
    fields = loadmat(BytesIO(bundle.read(member)))
    if "d_iner" not in fields:
        raise ValueError(f"Missing d_iner in {member}")
    return validate_signal(fields["d_iner"])


def load_trials(inertial, rgb=None, require_complete=True):
    index = archive_index(inertial, "inertial")
    if require_complete and len(index) != 861:
        raise ValueError(f"Expected 861 inertial trials; found {len(index)}")
    if require_complete:
        expected = {Trial(a, s, t) for a in range(1, 28) for s in range(1, 9)
                    for t in range(1, 5) if (a, s, t) not in MISSING_TRIALS}
        if set(index) != expected:
            raise ValueError("Trial IDs differ from the official 861-trial release")
    video = archive_index(rgb, "color") if rgb else None
    if video is not None and set(video) != set(index):
        missing = sorted(t.key for t in set(index) - set(video))
        extra = sorted(t.key for t in set(video) - set(index))
        raise ValueError(f"Video/IMU trial mismatch: missing={missing}, extra={extra}")
    signals = []
    rows = []
    with ZipFile(inertial) as bundle:
        for trial, member in index.items():
            signal = read_signal(bundle, member)
            signals.append(signal)
            rows.append({"trial": trial.key, "action": trial.action,
                         "subject": trial.subject, "repetition": trial.repetition,
                         "samples": len(signal), "inertial_member": member,
                         "video_member": video[trial] if video else None})
    return list(index), signals, rows


def subject_split(trials, train=DEVELOPMENT_SUBJECTS,
                  validation=VALIDATION_SUBJECTS, test=TEST_SUBJECTS):
    groups = [set(train), set(validation), set(test)]
    if any(not group for group in groups):
        raise ValueError("Subject groups must be non-empty")
    if any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Train, validation, and test subjects must not overlap")
    observed = {trial.subject for trial in trials}
    if observed != set.union(*groups):
        raise ValueError("Subject groups must exactly cover the data")
    splits = tuple(np.array([i for i, trial in enumerate(trials)
                            if trial.subject in group], dtype=int) for group in groups)
    actions = {trial.action for trial in trials}
    for indices in splits:
        if {trials[i].action for i in indices} != actions:
            raise ValueError("Every subject split must contain every action")
    return splits
