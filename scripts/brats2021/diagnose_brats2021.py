from pathlib import Path
import argparse, json
from prepare_brats2021 import choose_root, detect_label_scheme

ap=argparse.ArgumentParser()
ap.add_argument('--data-root', default='/datastore/phucngt/datasets/BraTS2021')
ap.add_argument('--label-scheme', default='auto')
ap.add_argument('--label-scan-max', type=int, default=128)
a=ap.parse_args()
root, rows, incomplete, nifti_count, layout, attempts=choose_root(Path(a.data_root).expanduser())
info = detect_label_scheme(rows, root, a.label_scheme, a.label_scan_max) if rows else None
print(json.dumps({
    'requested': a.data_root,
    'resolved': str(root),
    'layout': layout,
    'complete_cases': len(rows),
    'incomplete_groups': len(incomplete),
    'nifti_files_seen': nifti_count,
    'label_info': info,
    'attempts': attempts,
    'sample_case_ids': [r['case_id'] for r in rows[:10]],
}, indent=2))
