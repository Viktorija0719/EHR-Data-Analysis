# Electronic Health Record (EHR) Data Analysis

A small Python repository for building Table 1–style summary tables and automatically selecting statistical tests for continuous and categorical variables.


## Features

- Summary table generation by group
- Automatic statistical test selection based on:
  - variable type
  - number of groups
  - normality
  - variance equality
- Continuous variable support:
  - one-sample t-test
  - sign test
  - Student t-test
  - Welch t-test
  - Mann–Whitney U test
  - one-way ANOVA
  - Kruskal–Wallis test
  - optional post-hoc testing
- Categorical variable support:
  - Chi-square test
  - Fisher’s exact test for sparse 2×2 tables

## Repository structure

```text
scripts/      Reusable functions for summaries, testing, and table building
data/         Input and output data files
notebooks/    Jupyter notebooks for analysis
tests/        Optional unit tests
````

## Dataset

The example dataset used in this repository comes from Kaggle and can be downloaded with `kagglehub`:

```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("andrewmvd/heart-failure-clinical-data")

print("Path to dataset files:", path)
```

## Installation

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the analysis

Start JupyterLab from the repository directory:

```bash
jupyter lab
```

Then open:

```text
analysis.ipynb
```

1