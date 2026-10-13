from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

REGIONS = ("WT", "TC", "ET")
MODALITIES = ("t1", "t1ce", "t2", "flair")

# BraTS label conventions changed between legacy BraTS (e.g. 2021) and the
# continuous BraTS-GLI releases (2023+).  The medical regions are the same, but
# enhancing tumor (ET) moved from integer label 4 to label 3.
LABEL_SCHEMES = {
    "legacy_0124": {"et_label": 4, "allowed": {0, 1, 2, 4}},
    "gli_0123": {"et_label": 3, "allowed": {0, 1, 2, 3}},
}


def normalize_label_scheme(value: str | None) -> str:
    v = str(value or "").strip().lower().replace("-", "_")
    aliases = {
        "legacy": "legacy_0124", "legacy_0124": "legacy_0124",
        "brats2021": "legacy_0124", "2021": "legacy_0124", "0124": "legacy_0124",
        "gli": "gli_0123", "gli_0123": "gli_0123",
        "brats2023": "gli_0123", "2023": "gli_0123", "2024": "gli_0123",
        "0123": "gli_0123",
    }
    if v in aliases:
        return aliases[v]
    raise ValueError(f"Unknown BraTS label scheme: {value!r}. Expected legacy_0124 or gli_0123.")


def _load_nifti(path: str | Path) -> np.ndarray:
    try:
        import nibabel as nib
    except ImportError as exc:  # pragma: no cover - depends on runtime env
        raise ImportError("BraTS 2021 NIfTI loading requires nibabel. Install requirements.txt.") from exc
    return np.asarray(nib.load(str(path)).get_fdata(dtype=np.float32), dtype=np.float32)


def _zscore_nonzero(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    m = np.isfinite(x) & (x != 0)
    out = np.zeros_like(x, dtype=np.float32)
    if m.any():
        vals = x[m]
        mu = float(vals.mean())
        sd = max(float(vals.std()), 1e-6)
        out[m] = (vals - mu) / sd
        out = np.clip(out, -6.0, 6.0)
    return out


def labels_to_regions(
    seg: np.ndarray,
    label_scheme: str = "legacy_0124",
    *,
    validate: bool = True,
) -> np.ndarray:
    """Convert a BraTS integer label map to overlapping WT/TC/ET channels.

    Supported conventions
    ---------------------
    legacy_0124 : labels {0,1,2,4}; ET=4 (BraTS 2021-style)
    gli_0123    : labels {0,1,2,3}; ET=3 (BraTS-GLI 2023+-style)

    The label scheme is deliberately explicit here.  Per-case ``auto``
    inference is unsafe because a valid case can contain no ET voxels at all.
    Dataset-level auto-detection is therefore performed once when the manifest
    is created and the resolved scheme is stored in every manifest row.
    """
    scheme = normalize_label_scheme(label_scheme)
    spec = LABEL_SCHEMES[scheme]
    et_label = int(spec["et_label"])
    s = np.asarray(seg)
    if validate:
        if not np.isfinite(s).all():
            raise ValueError("Segmentation contains non-finite values")
        rounded = np.rint(s)
        if not np.allclose(s, rounded, atol=1e-5):
            raise ValueError("Segmentation contains non-integer label values")
        u = set(int(x) for x in np.unique(rounded))
        bad = sorted(u - set(spec["allowed"]))
        if bad:
            raise ValueError(
                f"Segmentation contains labels {sorted(u)} incompatible with "
                f"label_scheme={scheme}; unexpected labels={bad}. "
                "Check whether this is a BraTS 2021 (ET=4) or BraTS-GLI 2023+ (ET=3) release."
            )
    wt = np.isin(s, [1, 2, et_label])
    tc = np.isin(s, [1, et_label])
    et = s == et_label
    return np.stack([wt, tc, et], axis=0).astype(np.float32)


def enforce_nested_regions(mask: np.ndarray) -> np.ndarray:
    """BraTS nesting: ET subset TC subset WT."""
    m = np.asarray(mask).astype(bool).copy()
    if m.shape[0] != 3:
        raise ValueError(f"Expected [3,D,H,W], got {m.shape}")
    m[1] |= m[2]
    m[0] |= m[1]
    return m.astype(np.float32)


def resize_volume(x: np.ndarray, shape: tuple[int, int, int], is_mask: bool = False) -> np.ndarray:
    arr = torch.from_numpy(np.asarray(x, dtype=np.float32))[None]
    mode = "nearest" if is_mask else "trilinear"
    if mode == "nearest":
        out = F.interpolate(arr, size=shape, mode=mode)
    else:
        out = F.interpolate(arr, size=shape, mode=mode, align_corners=False)
    return out[0].numpy()


def _augment(image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Safe rigid augmentations preserving BraTS geometry.
    if torch.rand(()) < 0.5:
        image, mask = image.flip(-1), mask.flip(-1)
    if torch.rand(()) < 0.5:
        image, mask = image.flip(-2), mask.flip(-2)
    if torch.rand(()) < 0.25:
        image, mask = image.flip(-3), mask.flip(-3)
    if torch.rand(()) < 0.35:
        scale = 0.9 + 0.2 * torch.rand((image.shape[0], 1, 1, 1), dtype=image.dtype)
        bias = -0.1 + 0.2 * torch.rand((image.shape[0], 1, 1, 1), dtype=image.dtype)
        image = image * scale + bias
    return image, mask


class BraTS2021VolumeDataset(Dataset):
    """Patient-level 3D BraTS 2021 dataset.

    The manifest contains absolute or data-root-relative paths for four modalities and seg.
    A fixed patient-level split avoids slice leakage. All three BraTS regions are returned.
    """

    def __init__(
        self,
        manifest_csv: str | Path,
        split: str,
        data_root: str | Path,
        volume_shape: Iterable[int] = (128, 128, 128),
        train: bool = False,
        require_seg: bool = True,
    ):
        self.root = Path(data_root).resolve()
        self.df = pd.read_csv(manifest_csv)
        self.df = self.df[self.df["split"].astype(str) == str(split)].reset_index(drop=True)
        if self.df.empty:
            raise ValueError(f"No cases in split={split}")
        self.shape = tuple(int(v) for v in volume_shape)
        if len(self.shape) != 3:
            raise ValueError("volume_shape must have 3 integers")
        self.train = bool(train)
        self.require_seg = bool(require_seg)

    def __len__(self) -> int:
        return len(self.df)

    def _resolve(self, p: str) -> Path:
        q = Path(str(p))
        return q if q.is_absolute() else self.root / q

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        mods = []
        original_shape = None
        for mod in MODALITIES:
            raw = _load_nifti(self._resolve(row[f"{mod}_path"]))
            if original_shape is None:
                original_shape = tuple(int(v) for v in raw.shape)
            elif tuple(raw.shape) != original_shape:
                raise ValueError(f"Modality shape mismatch for case={row['case_id']}: {mod} has {raw.shape}, expected {original_shape}")
            x = _zscore_nonzero(raw)
            mods.append(x)
        image = np.stack(mods, axis=0)
        image = resize_volume(image, self.shape, is_mask=False)

        seg_path = str(row.get("seg_path", ""))
        if seg_path and seg_path.lower() != "nan":
            seg = _load_nifti(self._resolve(seg_path))
            # The manifest preparation step resolves the dataset-wide label
            # convention once.  This avoids the dangerous per-case ambiguity
            # where an ET-negative case contains only labels {0,1,2}.
            scheme_raw = row.get("label_scheme", "legacy_0124")
            scheme = normalize_label_scheme(scheme_raw)
            mask = resize_volume(labels_to_regions(seg, scheme, validate=True), self.shape, is_mask=True)
            mask = enforce_nested_regions(mask)
        elif self.require_seg:
            raise ValueError(f"Segmentation is required for split={row['split']} case={row['case_id']}")
        else:
            mask = np.zeros((3, *self.shape), dtype=np.float32)

        image_t = torch.from_numpy(np.ascontiguousarray(image)).float()
        mask_t = torch.from_numpy(np.ascontiguousarray(mask)).float()
        if self.train:
            image_t, mask_t = _augment(image_t, mask_t)
        # BraTS GLI releases are distributed in a co-registered standardized
        # space. After resizing, this scale lets HD95 be reported in approximate
        # physical mm for the standardized volumes used by this pipeline.
        effective_spacing_mm = tuple(float(o) / float(t) for o, t in zip(original_shape, self.shape))
        meta = {
            "case_id": str(row["case_id"]),
            "original_shape": original_shape,
            "effective_spacing_mm": effective_spacing_mm,
            "label_scheme": str(row.get("label_scheme", "legacy_0124")),
            "et_label": int(row.get("et_label", 4)),
        }
        return image_t, mask_t, meta


def case_split(
    case_ids: list[str],
    seed: int = 2021,
    source_val_fraction: float = 0.08,
    error_train_fraction: float = 0.12,
    controller_fraction: float = 0.12,
    policy_val_fraction: float = 0.08,
    test_fraction: float = 0.10,
) -> dict[str, str]:
    """Patient-level split with explicit meta-model separation.

    train: source segmentor training only
    source_val: source early stopping and error-map validation
    error_train: error-map fitting only
    controller: action gain/harm controller fitting only
    val: policy calibration only
    test: locked until every threshold and policy is frozen

    Keeping error_train separate from controller prevents an error-map model from
    generating overfit training features for the downstream action controller.
    """
    ids = np.asarray(sorted(set(map(str, case_ids))), dtype=object)
    rng = np.random.default_rng(int(seed))
    ids = ids[rng.permutation(len(ids))]
    n = len(ids)
    specs = [
        ("test", float(test_fraction)),
        ("val", float(policy_val_fraction)),
        ("controller", float(controller_fraction)),
        ("error_train", float(error_train_fraction)),
        ("source_val", float(source_val_fraction)),
    ]
    counts = [(name, max(1, int(round(n * frac)))) for name, frac in specs]
    if sum(v for _, v in counts) >= n:
        raise ValueError("Split fractions leave no source segmentor training cases")
    out: dict[str, str] = {}
    i = 0
    for name, count in counts:
        for cid in ids[i:i + count]:
            out[str(cid)] = name
        i += count
    for cid in ids[i:]:
        out[str(cid)] = "train"
    return out
