from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.brats2021.data import LABEL_SCHEMES, case_split, normalize_label_scheme

# BraTS 2021 official names are *_t1, *_t1ce, *_t2, *_flair, *_seg.
# Newer BraTS releases are also accepted so the scanner does not fail when a
# local archive has been normalized to t1n/t1c/t2w/t2f naming.
ALIASES = {
    "t1_path": ("_t1", "-t1", "_t1n", "-t1n", "_t1w", "-t1w"),
    "t1ce_path": ("_t1ce", "-t1ce", "_t1gd", "-t1gd", "_t1c", "-t1c"),
    "t2_path": ("_t2", "-t2", "_t2w", "-t2w"),
    "flair_path": ("_flair", "-flair", "_t2f", "-t2f"),
    "seg_path": ("_seg", "-seg", "_segmentation", "-segmentation", "_label", "-label"),
}
REQUIRED = ("t1_path", "t1ce_path", "t2_path", "flair_path", "seg_path")


def _is_nifti(p: Path) -> bool:
    n = p.name.lower()
    return n.endswith(".nii") or n.endswith(".nii.gz")


def _stem(p: Path) -> str:
    n = p.name
    if n.lower().endswith(".nii.gz"):
        return n[:-7]
    if n.lower().endswith(".nii"):
        return n[:-4]
    return p.stem


def _match_modality(p: Path):
    """Return (case_id, manifest_column) for a NIfTI filename, else None."""
    s = _stem(p)
    low = s.lower()
    exact = {
        "t1": "t1_path", "t1n": "t1_path", "t1w": "t1_path",
        "t1ce": "t1ce_path", "t1gd": "t1ce_path", "t1c": "t1ce_path",
        "t2": "t2_path", "t2w": "t2_path",
        "flair": "flair_path", "t2f": "flair_path",
        "seg": "seg_path", "segmentation": "seg_path", "label": "seg_path",
    }
    if low in exact:
        return p.parent.name, exact[low]
    # Longest aliases first avoids any accidental short-suffix ambiguity.
    pairs = sorted(
        ((alias, col) for col, aliases in ALIASES.items() for alias in aliases),
        key=lambda z: len(z[0]),
        reverse=True,
    )
    for alias, col in pairs:
        if low.endswith(alias):
            cid = s[: -len(alias)].rstrip("_-")
            # Also support folders containing generic names t1.nii.gz, etc.
            if not cid:
                cid = p.parent.name
            return cid, col
    return None


def _discover_named_nifti(root: Path):
    groups: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    nifti_count = 0
    for p in root.rglob("*"):
        if not p.is_file() or not _is_nifti(p):
            continue
        nifti_count += 1
        m = _match_modality(p)
        if m is None:
            continue
        cid, col = m
        # Keep cases separated by parent directory. This supports both one-case-
        # per-folder and flat folders containing many BraTS cases.
        key = (str(p.parent.resolve()), cid)
        old = groups[key].get(col)
        if old and Path(old).resolve() != p.resolve():
            raise RuntimeError(
                f"Ambiguous {col} for case={cid} under {p.parent}: {old} vs {p}"
            )
        groups[key][col] = str(p.resolve())

    rows = []
    incomplete = []
    for (_parent, cid), d in sorted(groups.items()):
        missing = [c for c in REQUIRED if c not in d]
        if not missing:
            rows.append({"case_id": cid, **{c: d[c] for c in REQUIRED}})
        else:
            incomplete.append((cid, d, missing))
    return rows, incomplete, nifti_count




def _seg_unique_labels(path: str | Path) -> set[int]:
    """Load one segmentation and return integer labels, with strict validation."""
    try:
        import nibabel as nib
    except ImportError as exc:
        raise ImportError("BraTS label-scheme detection requires nibabel. Install requirements.txt.") from exc
    img = nib.load(str(path))
    arr = np.asanyarray(img.dataobj)
    if not np.isfinite(arr).all():
        raise ValueError(f"Non-finite values found in segmentation: {path}")
    rounded = np.rint(arr)
    if not np.allclose(arr, rounded, atol=1e-5):
        raise ValueError(f"Segmentation is not integer-valued: {path}")
    return set(int(x) for x in np.unique(rounded))


def _looks_like_gli_2023(rows: list[dict[str, str]], root: Path) -> bool:
    """Filename-only fallback used only when sampled masks contain no ET voxels."""
    if "brats2023" in str(root).lower() or "brats-gli" in str(root).lower():
        return True
    for r in rows[:50]:
        cid = str(r.get("case_id", "")).lower()
        names = " ".join(Path(str(r.get(k, ""))).name.lower() for k in REQUIRED)
        if cid.startswith("brats-gli-") or any(tok in names for tok in ("-t1n.nii", "-t1c.nii", "-t2w.nii", "-t2f.nii")):
            return True
    return False


def detect_label_scheme(
    rows: list[dict[str, str]],
    root: Path,
    requested: str = "auto",
    scan_max: int = 128,
) -> dict:
    """Resolve dataset-wide BraTS label convention before any model is trained.

    Detection prioritizes actual segmentation integer labels.  A filename/layout
    heuristic is used only when scanned cases contain neither label 3 nor 4
    (possible if all sampled cases happen to be ET-negative).
    """
    req = str(requested or "auto").strip().lower()
    if req != "auto":
        scheme = normalize_label_scheme(req)
        spec = LABEL_SCHEMES[scheme]
        observed = set()
        checked = 0
        nscan = min(len(rows), max(1, int(scan_max)))
        idx = np.linspace(0, len(rows) - 1, num=nscan, dtype=int) if nscan < len(rows) else np.arange(len(rows))
        for i in idx:
            r = rows[int(i)]
            u = _seg_unique_labels(r["seg_path"]); observed |= u; checked += 1
            bad = u - set(spec["allowed"])
            if bad:
                raise RuntimeError(
                    f"Requested label_scheme={scheme}, but {r['seg_path']} contains incompatible labels {sorted(u)}. "
                    f"Unexpected labels: {sorted(bad)}"
                )
        return {"label_scheme": scheme, "et_label": int(spec["et_label"]),
                "observed_labels": sorted(observed), "checked_segmentations": checked,
                "detection_reason": "explicit_override"}

    observed: set[int] = set()
    checked = 0
    nscan = min(len(rows), max(1, int(scan_max)))
    # Spread the checks across the dataset instead of reading only the first N
    # cases. This makes accidental mixed conventions much easier to catch.
    if nscan == len(rows):
        sampled_rows = rows
    else:
        idx = np.linspace(0, len(rows) - 1, num=nscan, dtype=int)
        sampled_rows = [rows[int(i)] for i in idx]
    for r in sampled_rows:
        u = _seg_unique_labels(r["seg_path"]); observed |= u; checked += 1
        if 3 in observed and 4 in observed:
            raise RuntimeError(
                "Both labels 3 and 4 were observed across sampled segmentations. "
                "This suggests mixed BraTS label conventions or corrupted masks; refusing to continue."
            )

    if 4 in observed:
        scheme, reason = "legacy_0124", "observed_ET_label_4"
    elif 3 in observed:
        scheme, reason = "gli_0123", "observed_ET_label_3"
    else:
        if _looks_like_gli_2023(rows, root):
            scheme, reason = "gli_0123", "filename_layout_fallback_no_ET_seen"
        else:
            scheme, reason = "legacy_0124", "legacy_filename_fallback_no_ET_seen"

    spec = LABEL_SCHEMES[scheme]
    bad = observed - set(spec["allowed"])
    if bad:
        raise RuntimeError(
            f"Auto-detected label_scheme={scheme} but sampled masks contain incompatible labels {sorted(observed)}; "
            f"unexpected={sorted(bad)}"
        )
    return {"label_scheme": scheme, "et_label": int(spec["et_label"]),
            "observed_labels": sorted(observed), "checked_segmentations": checked,
            "detection_reason": reason}


def _read_channel_names(dataset_json: Path):
    if not dataset_json.exists():
        return None
    try:
        obj = json.loads(dataset_json.read_text(encoding="utf-8"))
    except Exception:
        return None
    raw = obj.get("channel_names") or obj.get("modality")
    if not isinstance(raw, dict):
        return None
    out = {}
    for k, v in raw.items():
        try:
            idx = int(k)
        except Exception:
            continue
        name = str(v).lower().replace("-", "").replace("_", "")
        if name in {"t1", "t1n", "t1w"}:
            out[idx] = "t1_path"
        elif name in {"t1ce", "t1c", "t1gd", "t1gad", "t1g"}:
            out[idx] = "t1ce_path"
        elif name in {"t2", "t2w"}:
            out[idx] = "t2_path"
        elif name in {"flair", "t2f"}:
            out[idx] = "flair_path"
    return out if len(out) >= 4 else None


def _discover_nnunet_layout(root: Path):
    """Support imagesTr/*_0000..0003 + labelsTr/* layout when present."""
    rows = []
    for images_tr in root.rglob("imagesTr"):
        if not images_tr.is_dir():
            continue
        labels_tr = images_tr.parent / "labelsTr"
        if not labels_tr.is_dir():
            continue
        mapping = _read_channel_names(images_tr.parent / "dataset.json")
        # Common BraTS order if a local conversion omitted dataset.json.
        if mapping is None:
            mapping = {0: "t1_path", 1: "t1ce_path", 2: "t2_path", 3: "flair_path"}
        groups = defaultdict(dict)
        pat = re.compile(r"^(.*)_([0-9]{4})$")
        for p in images_tr.iterdir():
            if not p.is_file() or not _is_nifti(p):
                continue
            m = pat.match(_stem(p))
            if not m:
                continue
            cid, ch = m.group(1), int(m.group(2))
            col = mapping.get(ch)
            if col:
                groups[cid][col] = str(p.resolve())
        labels = {}
        for p in labels_tr.iterdir():
            if p.is_file() and _is_nifti(p):
                labels[_stem(p)] = str(p.resolve())
        for cid, d in sorted(groups.items()):
            seg = labels.get(cid)
            if seg and all(c in d for c in ("t1_path", "t1ce_path", "t2_path", "flair_path")):
                rows.append({"case_id": cid, **d, "seg_path": seg})
    return rows


def discover(root: Path):
    named, incomplete, nifti_count = _discover_named_nifti(root)
    if named:
        return named, incomplete, nifti_count, "named_nifti"
    nnunet = _discover_nnunet_layout(root)
    if nnunet:
        return nnunet, incomplete, nifti_count, "nnunet_imagesTr"
    return [], incomplete, nifti_count, "none"


def candidate_roots(requested: Path):
    """Yield plausible BraTS2021 roots without requiring exact capitalization/name."""
    seen = set()

    def add(p: Path):
        try:
            q = p.resolve()
        except Exception:
            q = p.absolute()
        if q not in seen and q.exists() and q.is_dir():
            seen.add(q)
            return q
        return None

    q = add(requested)
    if q:
        yield q
        # Official downloads are often nested one folder below a user-created root.
        try:
            for child in requested.iterdir():
                if child.is_dir() and ("brats" in child.name.lower() or "training" in child.name.lower()):
                    q = add(child)
                    if q:
                        yield q
        except OSError:
            pass

    parent = requested.parent
    if parent.exists():
        try:
            siblings = sorted(
                [p for p in parent.iterdir() if p.is_dir() and "brats" in p.name.lower() and "2021" in p.name.lower()],
                key=lambda p: p.name.lower(),
            )
        except OSError:
            siblings = []
        for p in siblings:
            q = add(p)
            if q:
                yield q


def choose_root(requested: Path):
    attempts = []
    for root in candidate_roots(requested):
        rows, incomplete, nifti_count, layout = discover(root)
        attempts.append({
            "root": str(root),
            "complete_cases": len(rows),
            "incomplete_groups": len(incomplete),
            "nifti_files": int(nifti_count),
            "layout": layout,
        })
        if rows:
            return root, rows, incomplete, nifti_count, layout, attempts
    return requested, [], [], 0, "none", attempts


def _diagnostic_message(requested: Path, attempts):
    archive_ext = ("*.zip", "*.tar", "*.tar.gz", "*.tgz", "*.7z")
    archives = []
    if requested.exists():
        for pat in archive_ext:
            archives.extend(requested.rglob(pat))
            if len(archives) >= 10:
                break
    lines = [
        f"No labeled BraTS cases were discovered from requested root: {requested}",
        "The scanner now supports official BraTS2021 names, t1n/t1c/t2w/t2f aliases, nested roots, flat folders, and nnU-Net imagesTr/labelsTr layouts.",
        "Discovery attempts:",
    ]
    if attempts:
        for a in attempts:
            lines.append(
                f"  - {a['root']}: NIfTI={a['nifti_files']}, complete_cases={a['complete_cases']}, incomplete_groups={a['incomplete_groups']}, layout={a['layout']}"
            )
    else:
        lines.append("  - No existing candidate directory was found.")
    if archives:
        lines.append("Archives were found but are not auto-extracted for safety/storage reasons:")
        for p in archives[:10]:
            lines.append(f"  - {p}")
        lines.append("Extract the BraTS 2021 TrainingData archive, then resubmit the job.")
    lines += [
        "Expected official per-case example:",
        "  BraTS2021_00000/BraTS2021_00000_t1.nii.gz",
        "  BraTS2021_00000/BraTS2021_00000_t1ce.nii.gz",
        "  BraTS2021_00000/BraTS2021_00000_t2.nii.gz",
        "  BraTS2021_00000/BraTS2021_00000_flair.nii.gz",
        "  BraTS2021_00000/BraTS2021_00000_seg.nii.gz",
        "You may override the root at submission time, e.g.:",
        "  DATA_ROOT=/actual/path/to/BraTS2021_TrainingData sbatch run_uit.slurm",
    ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Create a patient-level BraTS 2021 manifest from labeled NIfTI data.")
    ap.add_argument("--data-root", default="/datastore/phucngt/datasets/BraTS2021")
    ap.add_argument("--output", default="data/brats2021_manifest.csv")
    ap.add_argument("--split-seed", type=int, default=2021)
    ap.add_argument("--source-val-fraction", type=float, default=0.08)
    ap.add_argument("--error-train-fraction", type=float, default=0.12)
    ap.add_argument("--controller-fraction", type=float, default=0.12)
    ap.add_argument("--policy-val-fraction", type=float, default=0.08)
    ap.add_argument("--test-fraction", type=float, default=0.10)
    ap.add_argument(
        "--label-scheme", default="auto",
        help="auto (recommended), legacy_0124/BraTS2021 (ET=4), or gli_0123/BraTS-GLI 2023+ (ET=3)",
    )
    ap.add_argument("--label-scan-max", type=int, default=128)
    args = ap.parse_args()

    requested = Path(args.data_root).expanduser()
    root, rows, incomplete, nifti_count, layout, attempts = choose_root(requested)
    if not rows:
        raise RuntimeError(_diagnostic_message(requested, attempts))

    # Deduplicate by case_id. Duplicate copies are dangerous because they can leak
    # the same patient into multiple meta-model cohorts.
    by_id = defaultdict(list)
    for r in rows:
        by_id[str(r["case_id"])].append(r)
    dups = {k: v for k, v in by_id.items() if len(v) > 1}
    if dups:
        sample = list(dups.items())[:5]
        raise RuntimeError(
            "Duplicate BraTS case IDs were found in multiple locations; refusing to split because this can cause patient leakage. "
            f"Examples: {[(k, [x['t1_path'] for x in v]) for k, v in sample]}"
        )
    labeled = [v[0] for v in by_id.values()]

    label_info = detect_label_scheme(
        labeled, root, requested=args.label_scheme, scan_max=args.label_scan_max
    )
    label_scheme = str(label_info["label_scheme"])
    et_label = int(label_info["et_label"])

    split = case_split(
        [r["case_id"] for r in labeled],
        args.split_seed,
        args.source_val_fraction,
        args.error_train_fraction,
        args.controller_fraction,
        args.policy_val_fraction,
        args.test_fraction,
    )
    for r in labeled:
        r["split"] = split[r["case_id"]]
        r["label_scheme"] = label_scheme
        r["et_label"] = et_label
        r["source_layout"] = layout
        # Absolute paths make fallback/autodetected roots safe for downstream stages.
        for col in REQUIRED:
            r[col] = str(Path(r[col]).resolve())

    df = pd.DataFrame(labeled).sort_values(["split", "case_id"]).reset_index(drop=True)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    meta = {
        "requested_data_root": str(requested),
        "resolved_data_root": str(root),
        "layout": layout,
        "n_cases": int(len(df)),
        "label_scheme": label_scheme,
        "et_label": et_label,
        "expected_integer_labels": sorted(LABEL_SCHEMES[label_scheme]["allowed"]),
        "observed_labels_during_detection": label_info["observed_labels"],
        "label_detection_reason": label_info["detection_reason"],
        "checked_segmentations_for_label_scheme": int(label_info["checked_segmentations"]),
        "split_counts": {str(k): int(v) for k, v in df["split"].value_counts().sort_index().items()},
        "manifest": str(out.resolve()),
    }
    meta_path = out.with_suffix(out.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("[BraTS data] requested_data_root:", requested)
    print("[BraTS data] resolved_data_root:", root)
    print("[BraTS data] layout:", layout)
    print("[BraTS data] n_cases:", len(df))
    print("[BraTS data] label_scheme:", label_scheme)
    print("[BraTS data] ET label:", et_label)
    print("[BraTS data] WT labels:", [1, 2, et_label])
    print("[BraTS data] TC labels:", [1, et_label])
    print("[BraTS data] observed labels during detection:", label_info["observed_labels"])
    print("[BraTS data] label detection reason:", label_info["detection_reason"])
    print("[BraTS data] split_counts:", meta["split_counts"])
    print("[BraTS data] manifest:", out)
    print("[BraTS data] metadata:", meta_path)
    counts = df.split.value_counts().to_dict()
    meta = {
        "requested_data_root": str(requested),
        "resolved_data_root": str(root),
        "manifest": str(out),
        "layout": layout,
        "nifti_files_seen": int(nifti_count),
        "n_cases": int(len(df)),
        "label_scheme": label_scheme,
        "et_label": et_label,
        "expected_integer_labels": sorted(LABEL_SCHEMES[label_scheme]["allowed"]),
        "observed_labels_during_detection": label_info["observed_labels"],
        "label_detection_reason": label_info["detection_reason"],
        "split_counts": {k: int(v) for k, v in counts.items()},
        "split_seed": args.split_seed,
        "source_val_fraction": args.source_val_fraction,
        "error_train_fraction": args.error_train_fraction,
        "controller_fraction": args.controller_fraction,
        "policy_val_fraction": args.policy_val_fraction,
        "test_fraction": args.test_fraction,
        "note": "Only labeled cases are used. Patient-level cohorts remain disjoint; official hidden BraTS test labels are not used.",
        "discovery_attempts": attempts,
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
