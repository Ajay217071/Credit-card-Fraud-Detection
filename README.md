# Dataset

The raw credit-card transaction dataset is intentionally **not committed to this repository** because it is large.

## Expected file

Place the Parquet dataset at:

```text
data/ieee_fraud_detection.parquet
```

The notebook loads the file from that location.

## Dataset structure

The project expects a transaction dataset containing the target column:

- `isFraud` — binary target where `0` represents a legitimate transaction and `1` represents fraud.

The notebook used the supplied Parquet dataset, which contains 590,528 transactions and 201 columns before preprocessing.

## Reproducing the project

Obtain the dataset from its original authorised source, save it locally using the filename above, install the dependencies from `requirements.txt`, and run the notebook in `notebooks/`.
