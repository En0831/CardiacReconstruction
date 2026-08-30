# utils/split.py

import argparse
import json
from datetime import datetime
from pathlib import Path
import numpy as np


def split_patients(data_root, seed=12345, train_ratio=0.6, valid_ratio=0.1):
    data_root = Path(data_root)

    patient_ids = sorted([p.name for p in data_root.glob("patient*") if p.is_dir()])

    if len(patient_ids) == 0:
        raise RuntimeError(f"No patient folders found in {data_root}")

    rng = np.random.default_rng(seed)
    patient_ids = np.array(patient_ids)
    rng.shuffle(patient_ids)

    n_total = len(patient_ids)
    n_train = int(train_ratio * n_total)
    n_valid = int(valid_ratio * n_total)

    train_ids = sorted(patient_ids[:n_train].tolist())
    valid_ids = sorted(patient_ids[n_train:n_train + n_valid].tolist())
    test_ids = sorted(patient_ids[n_train + n_valid:].tolist())

    return train_ids, valid_ids, test_ids


def load_split(split_json):
    split_json = Path(split_json)

    with open(split_json, "r") as f:
        split_info = json.load(f)

    for key in ["train", "valid", "test"]:
        if key not in split_info:
            raise KeyError(f"Split JSON is missing key '{key}': {split_json}")

    train_set = set(split_info["train"])
    valid_set = set(split_info["valid"])
    test_set = set(split_info["test"])

    if train_set & valid_set or train_set & test_set or valid_set & test_set:
        raise ValueError(f"Split JSON contains overlapping patient ids: {split_json}")

    return split_info


def main():
    parser = argparse.ArgumentParser(description="Create fixed CAMUS patient split JSON")

    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output", type=str, default="./splits/camus_split.json")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)

    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output file already exists: {output}")

    train_ids, valid_ids, test_ids = split_patients(
        data_root=args.data_root,
        seed=args.seed,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
    )

    split_info = {
        "meta": {
            "data_root": str(args.data_root),
            "seed": args.seed,
            "train_ratio": args.train_ratio,
            "valid_ratio": args.valid_ratio,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "n_train": len(train_ids),
            "n_valid": len(valid_ids),
            "n_test": len(test_ids),
        },
        "train": train_ids,
        "valid": valid_ids,
        "test": test_ids,
    }

    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w") as f:
        json.dump(split_info, f, indent=2)

    print(f"Saved split to {output}")
    print(f"Train patients: {len(train_ids)}")
    print(f"Valid patients: {len(valid_ids)}")
    print(f"Test patients:  {len(test_ids)}")


if __name__ == "__main__":
    main()