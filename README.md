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

### Baseline 2: Simple GNN
- Model: GraphSAGE / GAT
- Message passing on directed graph (optionally with reverse edges)
- Labeled nodes only for loss
- Class-weighted CrossEntropy loss
- Temporal split: train <= 34, val 35-41, test >= 42
 - Optional: self-loops, feature normalization, focal loss

#### Results

| Model | PR-AUC | Recall @ 1% FPR | Precision @ 1% |
|---|---:|---:|---:|
| XGBoost (features only) | 0.5554 | 0.4828 | 1.0000 |
| GraphSAGE (norm + reverse edges) | 0.3425 | 0.2402 | 0.8295 |

## 5. Next steps
- GNN training
- Temporal split evaluation
