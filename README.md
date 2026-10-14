# Exploratory Data Analysis: Auto MPG

following the [NIST/SEMATECH e-Handbook, Chapter 1 (EDA)](https://www.itl.nist.gov/div898/handbook/eda/eda.htm),
of the **Auto MPG** dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/9/auto+mpg),
which is listed in [Awesome Public Datasets](https://github.com/awesomedata/awesome-public-datasets).

**Notebook:** [`auto_mpg_eda.ipynb`](auto_mpg_eda.ipynb)


#
## Run it
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook auto_mpg_eda.ipynb
```
`build_notebook.py` generates the notebook from source; regenerate and run it with
`python build_notebook.py && jupyter nbconvert --to notebook --execute --inplace auto_mpg_eda.ipynb`.
