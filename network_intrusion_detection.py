#!/usr/bin/env python3
"""Network intrusion detection on NSL-KDD with a Random Forest.

Classifies a network connection as Normal or one of four attack families:
DoS, Probe, R2L (remote access) or U2R (privilege escalation).

Run (official train/test files, the recommended benchmark):
    python network_intrusion_detection.py --train KDDTrain+.txt --test KDDTest+.txt

Or with a single file (stratified random 80/20 split, much easier, so scores are optimistic):
    python network_intrusion_detection.py --data KDD+.txt

Add --report results.md to write the metrics as Markdown tables.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

RANDOM_STATE = 42

COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "attack", "level",  # label and difficulty level: never used as features
]
CATEGORICAL = ["protocol_type", "service", "flag"]
FEATURES = [c for c in COLUMNS if c not in ("attack", "level")]

CLASS_NAMES = ["Normal", "DoS", "Probe", "R2L", "U2R"]
ATTACK_FAMILIES = {
    "DoS": ["apache2", "back", "land", "mailbomb", "neptune", "pod", "processtable",
            "smurf", "teardrop", "udpstorm", "worm"],
    "Probe": ["ipsweep", "mscan", "nmap", "portsweep", "saint", "satan"],
    "R2L": ["ftp_write", "guess_passwd", "httptunnel", "imap", "multihop", "named", "phf",
            "sendmail", "snmpgetattack", "snmpguess", "spy", "warezclient", "warezmaster",
            "xlock", "xsnoop"],
    "U2R": ["buffer_overflow", "loadmodule", "perl", "ps", "rootkit", "sqlattack", "xterm"],
}
ATTACK_TO_CLASS = {"normal": 0}
for _name, _attacks in ATTACK_FAMILIES.items():
    for _attack in _attacks:
        ATTACK_TO_CLASS[_attack] = CLASS_NAMES.index(_name)


# --------------------------------------------------------------------------- data
def load_files(paths: list[Path]) -> pd.DataFrame:
    """Read one or more NSL-KDD files (no header, 43 columns) and add a `target` column."""
    frames = []
    for path in paths:
        frame = pd.read_csv(path, names=COLUMNS, index_col=False)
        if frame["attack"].isna().any():
            raise SystemExit(f"{path}: expected 43 comma-separated columns per row")
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)

    unknown = sorted(set(df["attack"]) - set(ATTACK_TO_CLASS))
    if unknown:  # never label unknown attacks as Normal by accident
        raise SystemExit(f"Attack names missing from the family mapping: {unknown}")
    df["target"] = df["attack"].map(ATTACK_TO_CLASS)
    return df


def describe(df: pd.DataFrame, title: str) -> None:
    counts = df["target"].map(dict(enumerate(CLASS_NAMES))).value_counts()
    print(f"{title}: {len(df)} connections")
    print(counts.reindex(CLASS_NAMES).to_string(), "\n")


# -------------------------------------------------------------------------- model
def build_model(balanced: bool) -> Pipeline:
    """One-hot encode the 3 categorical columns inside the pipeline, so the saved model
    accepts raw rows and ignores categories it has not seen during training."""
    encoder = ColumnTransformer(
        [("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)],
        remainder="passthrough",  # numeric and binary columns go through unchanged
    )
    forest = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced" if balanced else None,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline([("preprocess", encoder), ("classifier", forest)])


def top_features(model: Pipeline, n: int = 10) -> pd.Series:
    names = model.named_steps["preprocess"].get_feature_names_out()
    names = [name.split("__", 1)[-1] for name in names]
    importances = model.named_steps["classifier"].feature_importances_
    return pd.Series(importances, index=names).sort_values(ascending=False).head(n)


def markdown_report(y_true, y_pred) -> str:
    report = classification_report(
        y_true, y_pred, labels=range(len(CLASS_NAMES)), target_names=CLASS_NAMES,
        output_dict=True, zero_division=0,
    )
    lines = ["| Class | Precision | Recall | F1 | Support |", "|---|---|---|---|---|"]
    for name in CLASS_NAMES:
        r = report[name]
        lines.append(f"| {name} | {r['precision']:.2f} | {r['recall']:.2f} | "
                     f"{r['f1-score']:.2f} | {int(r['support'])} |")
    lines.append(f"| **Accuracy** | | | {report['accuracy']:.4f} | {len(y_true)} |")
    lines.append(f"| **Macro avg** | {report['macro avg']['precision']:.2f} | "
                 f"{report['macro avg']['recall']:.2f} | {report['macro avg']['f1-score']:.2f} | |")

    matrix = confusion_matrix(y_true, y_pred, labels=range(len(CLASS_NAMES)))
    lines += ["", "Confusion matrix (rows = actual, columns = predicted):", "",
              "| | " + " | ".join(CLASS_NAMES) + " |", "|---|" + "---|" * len(CLASS_NAMES)]
    for name, row in zip(CLASS_NAMES, matrix):
        lines.append(f"| **{name}** | " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description="NSL-KDD intrusion detection (Random Forest).")
    parser.add_argument("--train", type=Path, nargs="+", help="official training file(s), e.g. KDDTrain+.txt")
    parser.add_argument("--test", type=Path, nargs="+", help="official test file(s), e.g. KDDTest+.txt")
    parser.add_argument("--data", type=Path, nargs="+", help="single file(s) for a random 80/20 split")
    parser.add_argument("--balanced", action="store_true",
                        help="class_weight='balanced' (helps the rare classes, may cost precision)")
    parser.add_argument("--model-path", type=Path, default=Path("network_ids_model.joblib"))
    parser.add_argument("--report", type=Path, help="write the metrics as Markdown to this file")
    args = parser.parse_args()

    if args.train and args.test:
        train_df, test_df = load_files(args.train), load_files(args.test)
        mode = "official train/test files"
    elif args.data:
        full = load_files(args.data)
        train_df, test_df = train_test_split(
            full, test_size=0.2, stratify=full["target"], random_state=RANDOM_STATE
        )
        mode = "random stratified 80/20 split (optimistic)"
    else:
        parser.error("use either --train and --test, or --data")

    print(f"Evaluation mode: {mode}\n")
    describe(train_df, "Training set")
    describe(test_df, "Test set")

    model = build_model(args.balanced)
    model.fit(train_df[FEATURES], train_df["target"])

    predictions = model.predict(test_df[FEATURES])
    print(classification_report(
        test_df["target"], predictions, labels=range(len(CLASS_NAMES)),
        target_names=CLASS_NAMES, zero_division=0, digits=3,
    ))
    print("Confusion matrix (rows = actual, columns = predicted):")
    print(confusion_matrix(test_df["target"], predictions, labels=range(len(CLASS_NAMES))))
    macro = f1_score(test_df["target"], predictions, average="macro", zero_division=0)
    print(f"\nMacro F1: {macro:.3f}  (averages the 5 classes equally, so rare attacks count)")
    print("\nTop 10 features:")
    print(top_features(model).round(3).to_string())

    joblib.dump(model, args.model_path)
    print(f"\nModel saved to {args.model_path}")

    if args.report:
        args.report.write_text(
            f"Evaluation mode: {mode}\n\n" + markdown_report(test_df["target"], predictions) + "\n"
        )
        print(f"Metrics written to {args.report}")


if __name__ == "__main__":
    main()
