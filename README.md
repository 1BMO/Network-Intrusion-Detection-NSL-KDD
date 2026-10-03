# Network Intrusion Detection (NSL-KDD)

A small machine-learning project that classifies network connections as normal traffic or one of four attack families, built while studying offensive AI / adversarial machine learning. The goal is to understand how a detector trained on network flow features works end to end, and where it is weak, so I can later test how such models are evaded.

## Dataset

[NSL-KDD](https://www.unb.ca/cic/datasets/nsl.html), published by the Canadian Institute for Cybersecurity (University of New Brunswick). It is a cleaned-up version of the KDD Cup 1999 data: Tavallaee et al., "A Detailed Analysis of the KDD CUP 99 Data Set" (2009).

Each row is one network connection described by 41 features (duration, protocol, service, bytes sent and received, error rates, login indicators, ...), plus an attack name and a difficulty level. The files have no header row.

| File | Rows | Use |
|---|---|---|
| `KDDTrain+.txt` | 125,973 | training |
| `KDDTest+.txt` | 22,544 | testing |

The dataset is not included in this repository. Download it from the link above.

### Classes

The many attack names are grouped into four families:

| Class | Meaning | Examples |
|---|---|---|
| Normal | benign traffic | |
| DoS | denial of service (flooding) | neptune, smurf, teardrop |
| Probe | scanning and reconnaissance | nmap, portsweep, ipsweep |
| R2L | remote access without an account | guess_passwd, ftp_write, warezclient |
| U2R | local privilege escalation | buffer_overflow, rootkit, perl |

The script stops with an error if it meets an attack name that is not in the mapping, instead of silently labeling it Normal.

The classes are very imbalanced: U2R has only a few dozen training rows in the official training file, while Normal and DoS have tens of thousands.

## How it works

1. Read the files and map each attack name to its family.
2. One-hot encode `protocol_type`, `service` and `flag` inside the pipeline (categories not seen in training are ignored at prediction time). All other columns are used as they are; no scaling is needed for a Random Forest.
3. `attack` and `level` are never used as features, since they describe the label.
4. Train a `RandomForestClassifier` (100 trees, `random_state=42`).
5. Evaluate per class and save the whole pipeline with `joblib`.

## Run

```bash
git clone https://github.com/1BMO/Network-Intrusion-Detection-NSL-KDD.git
cd Network-Intrusion-Detection-NSL-KDD
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Recommended: the official train/test files
python network_intrusion_detection.py --train KDDTrain+.txt --test KDDTest+.txt
```

Options:

```bash
python network_intrusion_detection.py --data KDD+.txt          # one file, random stratified 80/20 split
python network_intrusion_detection.py --train KDDTrain+.txt --test KDDTest+.txt --balanced
python network_intrusion_detection.py --train KDDTrain+.txt --test KDDTest+.txt --report results.md
```

- `--balanced` uses `class_weight='balanced'`, which gives the rare classes more weight (usually better recall for them, sometimes at the cost of precision).
- `--report results.md` writes the metrics as Markdown tables.
- `--data` accepts a single file, for example `KDDTrain+.txt` and `KDDTest+.txt` joined into one file. The split is random, so train and test come from the same distribution and the scores are optimistic.

## Results

<!-- Run the script with --report results.md and paste the tables here. -->

Always judge the model by the per-class table and the macro average, not by overall accuracy. Normal and DoS make up most of the data, so accuracy can look excellent while a rare class such as U2R is mostly missed.

The official test file is intentionally harder than the training file (it contains attack types that do not appear in training), so scores on it are lower than on a random split of the same data.

## Using the saved model

The saved pipeline accepts raw rows, with the 41 original feature columns:

```python
from pathlib import Path

import joblib

from network_intrusion_detection import CLASS_NAMES, FEATURES, load_files

model = joblib.load("network_ids_model.joblib")
df = load_files([Path("KDDTest+.txt")])
predictions = model.predict(df[FEATURES].head(5))
print([CLASS_NAMES[p] for p in predictions])
```

> Only load `.joblib` files from sources you trust, since they can execute code when loaded.

## Limitations

- This is supervised classification: the model learns the attacks it was shown. It is not anomaly detection and will not reliably catch an attack type it has never seen.
- NSL-KDD is old and synthetic. It does not represent modern networks, encrypted traffic or current attack tools.
- The features are statistics computed by the dataset's authors, not raw packets, so the model cannot be used on live traffic without rebuilding those features.
- Rare classes (U2R, R2L) have few examples, so their scores are unstable.
- No hyperparameter tuning was done; the forest uses default settings.

## Next steps

- Compare with gradient boosting and a simple neural network.
- Tune hyperparameters with cross-validation on the training file only.
- Handle the class imbalance more carefully (resampling, per-class thresholds).
- Test how an attacker could change the features of a connection to evade the model, and document the effect.
