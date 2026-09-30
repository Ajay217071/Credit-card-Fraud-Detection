# Model Results

These figures are the results recorded in the original submitted notebook. They are included as a reference point for the refactored repository.

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Legitimate (0) | 0.98 | 0.99 | 0.99 | 113,952 |
| Fraud (1) | 0.69 | 0.52 | 0.59 | 4,154 |
| **Accuracy** | **0.97** | | | **118,106** |
| Macro average | 0.84 | 0.75 | 0.79 | 118,106 |
| Weighted average | 0.97 | 0.97 | 0.97 | 118,106 |

## Interpretation

The model performs strongly on the majority legitimate class, but fraud recall is 0.52 in the original run. This means the model did not identify a substantial share of fraudulent transactions.

These figures should be treated as the **original notebook benchmark**, not as a guarantee that a newly rerun refactored notebook will produce identical values. Preprocessing and library-version changes can affect model results.
