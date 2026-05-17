# Heart Disease Classification

This repository contains a small machine learning project that trains a binary classifier to predict heart disease from clinical patient attributes using the provided `heart.csv` dataset.

## Contents
- `heart.csv` — dataset with patient records
- `main.py` — training / evaluation script used for experiments
- `report.md` — project write-up and results

## Requirements
- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib (optional, for plots)

Install dependencies with pip:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not present, install directly:

```bash
pip install pandas numpy scikit-learn matplotlib
```

## Usage
Ensure `heart.csv` is in the repository root, then run:

```bash
python main.py
```

This will run the preprocessing, training, and evaluation pipeline described in `report.md`.

## Notes
- The preprocessing assumptions and feature engineering are documented in `report.md`.
- Adjust hyperparameters and model selection inside `main.py` as needed.

## License
This project is provided for educational purposes.
