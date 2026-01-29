# Transaction Classification (Elliptic Graph)

## 1. Overview

### Problem
Detect illicit (money laundering) transactions in the Bitcoin transaction graph.

### Why AML is hard
- Labels are sparse and noisy
- Illicit activity evolves over time
- Class imbalance is severe

### Why graphs
Transactions are linked; structure provides signal beyond tabular features.

## 2. Dataset

### Plain language description
- Nodes are transactions
- Directed edges represent money flow between transactions
- Many labels are missing because only a subset of transactions are investigated

### Simple diagram

```
Tx A  ->  Tx B  ->  Tx C
  \             /
   ->  Tx D  ->
```

## 3. Exploratory Data Analysis

Summary topics:
- Graph stats
- Label imbalance
- Temporal behavior
- Structural homophily

Notebook: `notebooks/01_eda.ipynb`

## 4. Baselines

### Baseline 1: Tabular (No Graph)
- Model: XGBoost on node features only
- Labeled nodes only
- Class-weighted loss
- Temporal split: train <= 34, val 35-41, test >= 42

#### Results

| Model | PR-AUC | Recall @ 1% FPR | Precision @ 1% |
|---|---:|---:|---:|
| XGBoost (features only) | TBD | TBD | TBD |

## 5. Next steps
- GNN training
- Temporal split evaluation
