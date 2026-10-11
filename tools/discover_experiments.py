import json
import glob
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
exp_pattern = str(BASE_DIR / "experiments" / "**" / "metrics.json")

print("=" * 60)
print("DISCOVERING ALL EXPERIMENTS METRICS")
print("=" * 60)

for p in sorted(glob.glob(exp_pattern, recursive=True)):
    path_obj = Path(p)
    try:
        with open(path_obj) as f:
            d = json.load(f)
            print(f"\nFile: {path_obj}")
            if "accuracy" in d:
                print(f"  Accuracy: {d.get('accuracy'):.4f}")
                print(f"  Precision: {d.get('precision'):.4f}")
                print(f"  Recall: {d.get('recall'):.4f}")
                print(f"  F1: {d.get('f1'):.4f}")
                print(f"  ROC-AUC: {d.get('auc', d.get('roc_auc'))}")
                print(f"  Balanced Accuracy: {d.get('balanced_accuracy')}")
                print(f"  Real F1: {d.get('f1_real')}")
                print(f"  Fake F1: {d.get('f1_fake')}")
            else:
                print(f"  Top Keys: {list(d.keys())}")
    except Exception as e:
        print(f"  Error loading {p}: {e}")
