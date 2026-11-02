# Data Mining & Warehousing (DMW) — Activity 2
## Socioeconomic Profiling and Income Prediction Using the Adult Census Dataset

### Project Overview
This repository contains the complete implementation for **DMW Activity 2**. The study focuses on socioeconomic profiling, data warehousing, association analysis, classification, and unsupervised clustering over the canonical **Adult Census Income dataset** (1994 U.S. Census Bureau extraction by Ronny Kohavi and Barry Becker).

The primary analytical objective is predicting whether an individual earns more than **$50,000 annually** (`>50K` vs. `<=50K`), exploring structural demographic disparities, designing an enterprise multidimensional data warehouse with OLAP queries, and evaluating data mining algorithms under strict methodological standards.

---

### Project Structure

```text
adult-income-mining/
├── adult/                                    # Untouched original UCI dataset files
│   ├── Index                                 # Archive index
│   ├── adult.data                            # Official training set (32,561 records)
│   ├── adult.names                           # Official data dictionary & benchmark doc
│   ├── adult.test                            # Official test set (16,281 records)
│   └── old.adult.names                       # Legacy UCI metadata archive
├── data/
│   ├── raw/                                  # Non-duplicative symlinks to original dataset
│   │   ├── adult.data -> ../../adult/adult.data
│   │   ├── adult.names -> ../../adult/adult.names
│   │   └── adult.test -> ../../adult/adult.test
│   └── processed/                            # Preprocessed, cleaned datasets
│       ├── adult_cleaned_train.csv           # Cleaned training data (32,561 x 13)
│       └── adult_cleaned_test.csv            # Cleaned test data (16,281 x 13)
├── notebooks/                                # Executed Jupyter Notebooks with full outputs
│   ├── 01_eda_and_similarity.ipynb           # EDA, 5-number summaries, skewness & similarity
│   ├── 02_preprocessing_pca_warehouse.ipynb  # Preprocessing, PCA & SQLite Star Schema OLAP
│   ├── 03_classification_and_association.ipynb # Decision Tree, RIPPER, KNN & Association Rules
│   └── 04_clustering.ipynb                   # K-Means, Hierarchical Dendrogram & DBSCAN
├── warehouse/                                # Relational OLAP Data Warehouse
│   ├── schema.sql                            # DDL script creating Star Schema tables & indexes
│   ├── olap_queries.sql                      # SQL scripts for 4 core OLAP operations
│   └── adult_dw.db                           # Local SQLite database populated with data
├── outputs/                                  # High-resolution saved plots & visualizations
│   ├── eda_numeric_distributions.png         # Numeric distributions and skewness profiles
│   ├── eda_categorical_counts.png            # Frequency distributions of primary categoricals
│   ├── eda_income_distribution.png           # Target income class imbalance bar chart
│   ├── profile_similarity_heatmaps.png       # Euclidean, Manhattan & Cosine heatmaps
│   ├── scaling_comparison.png                # Min-Max vs Z-Score comparison KDEs
│   ├── pca_scree_plot.png                    # Scree plot & cumulative variance curve
│   ├── pca_2d_projection.png                 # 2D PCA latent space colored by income
│   ├── decision_tree_structure.png           # Pruned decision tree architecture
│   ├── tree_feature_importances.png          # Top 10 Gini feature importances
│   ├── classification_roc_curves.png         # Decision Tree vs KNN ROC curves
│   ├── classification_confusion_matrices.png # Confusion matrices on held-out test data
│   ├── association_rules_comparison.png      # Apriori vs FP-Growth runtime & rule scatter
│   ├── kmeans_elbow_silhouette.png           # Elbow curve & silhouette scores for k=2..6
│   ├── hierarchical_dendrogram.png           # Ward's minimum variance dendrogram (N=750)
│   ├── dbscan_k_distance.png                 # 5-NN distance graph for eps radius detection
│   └── clustering_cluster_comparisons.png    # 2D cluster projections vs true income labels
├── scripts/                                  # Notebook generation & verification scripts
│   ├── data_utils.py                         # Ingestion & data-hygiene utilities
│   ├── generate_notebook_1.py
│   ├── generate_notebook_2.py
│   ├── generate_notebook_3.py
│   └── generate_notebook_4.py
├── run_project.py                            # End-to-end master execution runner
├── requirements.txt                          # Pinned project dependencies
└── README.md                                 # Comprehensive project report & guide
```

---

### Dataset Placement (Local Setup)

The original raw dataset files (`adult.data`, `adult.test`, `adult.names`) are excluded from this Git repository to preserve local data privacy and comply with repository size best practices.

To run the project on a fresh clone:
1. Download the Adult Census Income dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/2/adult) or extract the course-provided archive.
2. Place the uncompressed files into either:
   - `adult/` directory (`adult/adult.data`, `adult/adult.test`, `adult/adult.names`), or
   - `data/raw/` directory (`data/raw/adult.data`, `data/raw/adult.test`, `data/raw/adult.names`).
3. The project scripts and notebooks automatically detect and load the dataset from either location.

---

### Setup & Installation Instructions

#### Prerequisites
- Python 3.10+ (tested on Python 3.14)
- Git and standard terminal shell

#### 1. Environment Setup
```bash
# Navigate to project root
cd adult-income-mining

# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
# On macOS / Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### 2. Execute the Entire Workflow
To execute all four notebooks sequentially, populate the SQLite warehouse, run all OLAP queries, and regenerate all high-resolution figures in `outputs/`:
```bash
python run_project.py
```

Alternatively, open and run the notebooks interactively via Jupyter:
```bash
jupyter notebook notebooks/
```

---

### Notebook Details & Technical Methodology

#### Notebook 1: Exploratory Data Analysis & Profile Similarity (`01_eda_and_similarity.ipynb`)
- **Ingestion Hygiene:** Handles comma-space delimiters (`skipinitialspace=True`), missing markers (`na_values='?'`), skips comment row 1 on `adult.test`, strips trailing periods from test labels (`<=50K.`), and preserves original dataset immutability.
- **Descriptive Statistics:** Generates comprehensive five-number summaries, mean, standard deviation, IQR, skewness, and kurtosis for all 6 numeric variables.
- **Distribution & Anomaly Checks:** Discovers severe positive skewness in `capital-gain` (skew = 11.95) and `capital-loss` (skew = 4.59), where >91% of values are zero, and notes upper-bound censoring at $99,999.
- **Missing Value Co-occurrence:** Demonstrates that when `workclass` is missing, `occupation` is missing 99.8% of the time, identifying non-civilian or non-working individuals.
- **Profile Similarity Analysis:** Constructs 5 distinct demographic personas. After standardization of continuous features and one-hot encoding of categorical features, pairwise distance matrices are computed:
  - *Euclidean Distance:* Penalizes large discrepancies across continuous dimensions quadratically.
  - *Manhattan Distance:* Less sensitive to extreme coordinate spikes; highly robust over sparse one-hot spaces.
  - *Cosine Similarity:* Measures directional alignment in multidimensional space invariant to vector magnitude, yielding high similarity between profiles sharing occupational roles regardless of scale.

---

#### Notebook 2: Preprocessing, PCA & Data Warehousing (`02_preprocessing_pca_warehouse.ipynb`)
- **Feature Elimination Rationale:**
  - *Dropping `fnlwgt`:* Census sampling weight computed from state-level non-institutional demographic ratios. Irrelevant to individual earning power and risks fitting to state sampling artifacts.
  - *Dropping `education`:* `education-num` provides an exact monotonic integer mapping ($1 = \text{Preschool}$ to $16 = \text{Doctorate}$). Dropping `education` eliminates redundant multicollinearity without information loss.
- **Scaling Comparison:** Formulates and visualizes Min-Max Normalization versus Z-Score Standardization. Explains why Z-score is mathematically superior when extreme outliers exist (such as $99,999 capital gain compressing all other Min-Max values to $[0.00, 0.02]$).
- **Principal Component Analysis (PCA):**
  - **PC1 (25.93% variance):** Socioeconomic human capital index (loads positively on `education-num`, `hours-per-week`, `capital-gain`, and `age`).
  - **PC2 (20.60% variance):** Capital market activity divergence (loads on `capital-loss` vs `capital-gain`).
  - **PC3 (19.48% variance):** Career stage / maturity (loads on `age` vs `education-num`).
  - **PC4 (17.75% variance):** Work commitment / overtime devotion (loads on `hours-per-week`).
  - 4 components retain **83.77%** of total variance.
- **Data Warehouse (Star Schema in SQLite):**
  - `FactIncome`: Central fact table recording `capital_gain`, `capital_loss`, `net_capital`, `hours_per_week`, `income_bracket`, and binary flag `is_high_income`.
  - Dimension Tables: `DimDemographics`, `DimEducation`, `DimOccupation`, `DimGeography`.
  - Four OLAP Operations:
    1. *Roll-up:* Monotonic income progression across education tiers (Doctorate: 74.1% high-income; HS-grad: 15.9%; <1st grade: 0%).
    2. *Drill-down:* Breakdown of degree holders by occupation and gender.
    3. *Slice & Dice:* Regional slice (U.S.) diced across age cohorts and employment sectors.
    4. *Pivot / Cross-Tabulation:* High-income percentages across workclasses and marital status.

---

#### Notebook 3: Classification & Association Rules (`03_classification_and_association.ipynb`)
- **Decision Tree Classifier:**
  - Pruned tree (`max_depth=5`, `min_samples_leaf=25`) fitted within a `Pipeline` to prevent target leakage.
  - Gini feature importances reveal `marital-status_Married-civ-spouse`, `capital-gain`, and `education-num` as top predictors.
  - Test Accuracy: **85.34%**, Test ROC-AUC: **0.893**, Test F1 on `>50K`: **0.638**.
- **Rule-Based Classification (RIPPER):**
  - Implemented using `wittgenstein`.
  - Discovers compact propositional rules (e.g., `[marital-status=Married-civ-spouse ^ education-num>=13] => >50K`).
  - Test Accuracy: **83.1%**.
- **K-Nearest Neighbors ($k=5$):**
  - Evaluated with distance weighting inside a strict preprocessing pipeline.
  - Test Accuracy: **82.35%**, Test ROC-AUC: **0.854**, Test F1 on `>50K`: **0.612**.
- **Validation & Class Imbalance:**
  - Stratified 5-Fold Cross-Validation confirms model stability across training partitions.
  - Analysis highlights that predicting the majority class yields 76% accuracy but 0% recall on high earners; our models achieve balanced precision (~76%) and recall (~55-58%) on the minority positive class.
- **Association Rule Mining (Apriori vs. FP-Growth):**
  - Binned numeric attributes (`age`, `hours-per-week`, `capital-gain`, `education-num`).
  - Evaluated minimum support $s = 0.10$ and minimum confidence $c = 0.50$.
  - FP-Growth completes significantly faster than Apriori by building a compact FP-Tree without generating candidate $k$-itemsets.
  - Discovers high-lift rules predicting income class (e.g., `Age:<30 ^ NotMarried => <=50K` with lift 3.36).

---

#### Notebook 4: Unsupervised Clustering (`04_clustering.ipynb`)
- **K-Means Clustering:**
  - Evaluated $k \in \{2, 3, 4, 5, 6\}$ using inertia (WCSS) and silhouette analysis.
  - $k=3$ emerges as the optimal compromise:
    - *Cluster 0 (Entry-level Workforce):* Mean age 29, HS-grad level, low capital gain, 12% high-income rate.
    - *Cluster 1 (Prime Career / Educated):* Mean age 45, college/bachelors, 44% high-income rate.
    - *Cluster 2 (Asset Accumulators):* Substantial capital gains/losses ($>$ $8,000 avg), 62% high-income rate.
- **Hierarchical Agglomerative Clustering:**
  - Applied Ward's minimum variance linkage on a representative stratified sample ($N=750$).
  - Dendrogram confirms natural major divergence separating asset accumulators from standard wage workers.
- **DBSCAN:**
  - Parameter selection grounded in a 5-NN distance graph elbow at $\varepsilon = 0.65$, $\text{MinPts} = 10$.
  - Identifies a dense core manifold comprising ~94% of individuals while isolating ~4% as statistical outliers/noise points.
- **Cross-Tabulation Against Ground Truth:**
  - Compares unsupervised cluster assignments with true income brackets, demonstrating that unsupervised partitioning naturally aligns along human capital and asset-ownership divides.

---

### Empirical Metrics Benchmark Table

| Model / Evaluation Phase | Accuracy | Precision (`>50K`) | Recall (`>50K`) | F1-Score (`>50K`) | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Zero-Rule Baseline (Majority Class)** | 76.38% | 0.00% | 0.00% | 0.00% | 0.500 |
| **Decision Tree (`max_depth=5`)** | **85.34%** | **76.92%** | **54.55%** | **0.6381** | **0.893** |
| **KNN (`k=5`, distance-weighted)** | **82.35%** | **65.80%** | **57.18%** | **0.6119** | **0.854** |
| **RIPPER Rule Classifier** | **83.10%** | **71.20%** | **51.40%** | **0.5973** | — |
| **Stratified 5-Fold CV (Decision Tree)** | 85.22 ± 0.004 | 77.41 ± 0.008 | 53.68 ± 0.009 | 0.6340 ± 0.006 | 0.891 ± 0.003 |
| **Stratified 5-Fold CV (KNN k=5)** | 82.16 ± 0.004 | 65.42 ± 0.007 | 56.84 ± 0.008 | 0.6082 ± 0.006 | 0.851 ± 0.004 |

---

### Authors & Academic Integrity
- **Course:** Data Mining & Warehousing (DMW)
- **Assignment:** Activity 2 — Socioeconomic Profiling and Income Prediction
- All original dataset files have been strictly preserved in their unaltered form.

