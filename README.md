# Exploratory Data Analysis: Auto MPG

A thorough EDA, following the [NIST/SEMATECH e-Handbook, Chapter 1 (EDA)](https://www.itl.nist.gov/div898/handbook/eda/eda.htm),
of the **Auto MPG** dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/9/auto+mpg),
which is listed in [Awesome Public Datasets](https://github.com/awesomedata/awesome-public-datasets).

**Notebook:** [`auto_mpg_eda.ipynb`](auto_mpg_eda.ipynb)

## Dataset
398 cars (model years 1970–1982) × 9 attributes:
- **Numerical (6):** `mpg`, `displacement`, `horsepower`, `weight`, `acceleration`, `model_year`
- **Categorical (3):** `cylinders`, `origin` (USA/Europe/Japan), `car_name` (→ `make`)

## Contents
1. Loading and cleaning (missing horsepower, make-name spelling errors)
2. Summary statistics with 1.5×IQR outlier detection
3. A histogram for every numerical variable (with mean, median and outliers marked) and a bar chart for every categorical variable, each with a written report of min / max / mean / median / outliers
4. All variables side by side in a single box plot (z-scores, plus raw values on a log scale)
5. NIST 4-plot (run sequence, lag, histogram, normal probability) for `mpg`
6. Scatter plot matrix and correlation heatmap for the numerical variables
7. Categorical vs categorical: cross-tab heatmap, 100% stacked bars, chi-square / Cramér's V, and aggregation (pivot tables of mean mpg by origin × cylinders and by year × origin)
8. Categorical vs numerical: scatter matrix colored by origin, box/strip/violin plots by group, and a scatter with color = origin and size = cylinders
9. Conclusions

## Run it
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook auto_mpg_eda.ipynb
```
`build_notebook.py` generates the notebook from source; regenerate and run it with
`python build_notebook.py && jupyter nbconvert --to notebook --execute --inplace auto_mpg_eda.ipynb`.
