# Machine Learning I - Pedagogical Publication

Hands-on materials for an undergraduate machine-learning course, combining written exercises with Python experiments. The activities develop an understanding of how models learn, how their assumptions shape predictions, and how to interpret their performance.

## Hands-On contents

| Hands-On | Topics | Notebook activities |
|---|---|---|
| [HO01](hands-on/ho01/) | Environment and data exploration | [Explore data](hands-on/ho01/notebook.ipynb): inspect, filter and group observations; compare distributions. |
| [HO02](hands-on/ho02/) | Decision trees and evaluation | [Investigate trees](hands-on/ho02/notebook.ipynb): trace decisions, inspect errors and vary tree capacity. |
| [HO03](hands-on/ho03/) | k-nearest neighbours and Naive Bayes | [Explore neighbourhoods](hands-on/ho03/notebook.ipynb): change neighbour counts and weights; compare decision boundaries. |
| [HO04](hands-on/ho04/) | Neural networks and support vector machines | [Compare model geometry](hands-on/ho04/notebook.ipynb): explore kernels, margins, network capacity and initialisation. |
| [HO05](hands-on/ho05/) | Linear and tree regression | [Compare regression models](hands-on/ho05/notebook.ipynb): fit lines and surfaces, measure errors and inspect residuals. |
| [HO06](hands-on/ho06/) | Empirical evaluation and algorithm comparison | [Compare performance](hands-on/ho06/notebook.ipynb): examine cross-validation variability, dataset-level results and ranks. |
| [HO07](hands-on/ho07/) | Clustering | [Explore partitions](hands-on/ho07/notebook.ipynb): compare K-Means, DBSCAN and hierarchical clustering; investigate scaling and silhouette scores. |
| [HO08](hands-on/ho08/) | Ensembles and imbalanced classification | [Investigate imbalance](hands-on/ho08/notebook.ipynb): compare metrics, class weights and decision thresholds. |
| [HO09](hands-on/ho09/) | Meta-learning and algorithm selection | [Learn across datasets](hands-on/ho09/notebook.ipynb): relate dataset characteristics to algorithm performance and examine a simple recommendation model. |
| [HO10](hands-on/ho10/) | Association rules | Handout exercises on itemsets, support, confidence and lift. |

## Handouts and worked solutions

**Pre-rendered PDF and DOCX files are included and ready to open. Use PDF for reading or printing, DOCX for Word, and QMD to view or edit the source that generates both formats.**

| Hands-On | Handout contents | Handout | Worked solutions |
|---|---|---|---|
| HO02 | CART decision trees: Gini impurity, candidate splits, recursive partitioning and decision rules. | [PDF](hands-on/ho02/handout.pdf) · [DOCX](hands-on/ho02/handout.docx) · [QMD](hands-on/ho02/handout.qmd) | [PDF](hands-on/ho02/solutions/handout.pdf) · [DOCX](hands-on/ho02/solutions/handout.docx) · [QMD](hands-on/ho02/solutions/handout.qmd) |
| HO03 | Numerical and categorical k-NN, Manhattan distance, decision boundaries, and a guided Naive Bayes calculation. | [PDF](hands-on/ho03/handout.pdf) · [DOCX](hands-on/ho03/handout.docx) · [QMD](hands-on/ho03/handout.qmd) | [PDF](hands-on/ho03/solutions/handout.pdf) · [DOCX](hands-on/ho03/solutions/handout.docx) · [QMD](hands-on/ho03/solutions/handout.qmd) |
| HO04 | Single-neuron and two-layer networks: forward passes, loss, the chain rule, backpropagation, weight updates, and XOR. | [PDF](hands-on/ho04/handout.pdf) · [DOCX](hands-on/ho04/handout.docx) · [QMD](hands-on/ho04/handout.qmd) | [PDF](hands-on/ho04/solutions/handout.pdf) · [DOCX](hands-on/ho04/solutions/handout.docx) · [QMD](hands-on/ho04/solutions/handout.qmd) |
| HO05 | Linear regression calculations: slope and intercept, predictions, residuals, MSE, and the effect of changing an observation. | [PDF](hands-on/ho05/handout.pdf) · [DOCX](hands-on/ho05/handout.docx) · [QMD](hands-on/ho05/handout.qmd) | [PDF](hands-on/ho05/solutions/handout.pdf) · [DOCX](hands-on/ho05/solutions/handout.docx) · [QMD](hands-on/ho05/solutions/handout.qmd) |
| HO06 | Shared stratified cross-validation, mean accuracy and variability, score distributions, and dataset-level ranks and the Friedman test. | [PDF](hands-on/ho06/handout.pdf) · [DOCX](hands-on/ho06/handout.docx) · [QMD](hands-on/ho06/handout.qmd) | [PDF](hands-on/ho06/solutions/handout.pdf) · [DOCX](hands-on/ho06/solutions/handout.docx) · [QMD](hands-on/ho06/solutions/handout.qmd) |
| HO07 | Manual clustering of eight points: K-Means assignments and centroids, DBSCAN point types, single-link merges, and a dendrogram. | [PDF](hands-on/ho07/handout.pdf) · [DOCX](hands-on/ho07/handout.docx) · [QMD](hands-on/ho07/handout.qmd) | [PDF](hands-on/ho07/solutions/handout.pdf) · [DOCX](hands-on/ho07/solutions/handout.docx) · [QMD](hands-on/ho07/solutions/handout.qmd) |
| HO08 | Classifier ensembles and majority voting. Confusion matrices, accuracy, precision, recall, and F1 under class imbalance. | [PDF](hands-on/ho08/handout.pdf) · [DOCX](hands-on/ho08/handout.docx) · [QMD](hands-on/ho08/handout.qmd) | [PDF](hands-on/ho08/solutions/handout.pdf) · [DOCX](hands-on/ho08/solutions/handout.docx) · [QMD](hands-on/ho08/solutions/handout.qmd) |
| HO09 | Constructing dataset-level meta-features. Comparing a tree and a stump. Recommending an algorithm using 1-NN and 3-NN meta-learning. | [PDF](hands-on/ho09/handout.pdf) · [DOCX](hands-on/ho09/handout.docx) · [QMD](hands-on/ho09/handout.qmd) | [PDF](hands-on/ho09/solutions/handout.pdf) · [DOCX](hands-on/ho09/solutions/handout.docx) · [QMD](hands-on/ho09/solutions/handout.qmd) |
| HO10 | Transactions, frequent itemsets, support, confidence, lift, and the interpretation of association rules. | [PDF](hands-on/ho10/handout.pdf) · [DOCX](hands-on/ho10/handout.docx) · [QMD](hands-on/ho10/handout.qmd) | [PDF](hands-on/ho10/solutions/handout.pdf) · [DOCX](hands-on/ho10/solutions/handout.docx) · [QMD](hands-on/ho10/solutions/handout.qmd) |

To change a handout, edit its QMD source and follow [Regenerate the handouts](#regenerate-the-handouts) below.

## Using the materials

**Notebooks are available for HO01–HO09.** Each contains runnable experiments, plots, and short questions. Students should record a prediction before running an experiment, change the relevant parameters, and explain what the results reveal about the model. This shifts the lesson's focus from coding to an intuitive, empirical analysis of machine learning concepts.

Each student notebook has a matching `solutions/notebook.ipynb` with the same experiments and concise worked answers. Run cells in order to reproduce the results and use the interactive controls.

The data used by the notebooks are included in [data/](data/).

## Repository layout

```text
/
├── hands-on/                  Teaching sources, solutions and figures
│   ├── ho01/                  Student and solution notebooks only
│   ├── ho02/ ... ho09/        Notebooks and written handouts
│   │   ├── notebook.ipynb
│   │   ├── handout.qmd
│   │   ├── media/             Referenced figures, where needed
│   │   └── solutions/         Matching notebooks and handouts
│   └── ho10/                  Handout and worked solution only
├── data/                      Supplied datasets and attribution
├── src/mlcourse/              Shared notebook tools
├── templates/                 PDF and Word publishing styles
├── scripts/
│   ├── build_materials.py     Generate PDF/DOCX handouts
│   ├── check_notebooks.py     Execute any or all notebooks
│   ├── validate_materials.py  Check notebook pairs, data, and image references
│   └── prepare_data.py        Fetch or reproduce teaching datasets
├── tests/
│   ├── test_notebooks.py      Setup, notebook structure, and interactive tools
│   └── test_data.py           Dataset integrity and reproducibility
├── .github/                   Automated checks
├── _quarto.yml                Handout publishing settings
├── environment.yml            Conda environment
├── requirements.txt           Shared course packages
├── pyproject.toml             Course package installation
├── LICENSE.md                 Licence scope and attribution
├── LICENSES/                  Full CC BY 4.0 and MIT licence texts
└── .gitignore                 Local/generated-file exclusions
```

## Licence

Teaching materials are available under **CC BY 4.0**, and code under **MIT**. You can reuse and adapt the course with the required attribution and notices. Third-party datasets retain their own terms. See [licence and attribution](LICENSE.md) for the scope and full licence texts.

## Requirements and commands

Run the following commands from the repository folder.

### Open the notebooks

Install Conda or Miniconda, then create and activate the course environment:

```bash
conda env create -f environment.yml
conda activate ml-course
python -m pip install -e .
jupyter lab
```

This prepares Python and the required packages, then opens JupyterLab. If you use an IDE with notebook support, select the `ml-course` environment there and open the notebooks directly instead. Choose a `notebook.ipynb` inside `hands-on/` to begin. On later visits, activate `ml-course` and run `jupyter lab` (if applicable, otherwise just open the notebooks).

### Regenerate the handouts

**The PDF and DOCX files are already included. Rebuilding is only needed when modifying the materials.** Edit the `.qmd` sources (rebuilding replaces the generated PDF/DOCX files).

#### Required tools

1. Create the course environment as shown above and activate it with `conda activate ml-course`.
2. **Install [Quarto](https://quarto.org/docs/get-started/).** The Python environment does not install it.
3. **Install LaTeX with LuaLaTeX support** for PDF generation. If you do not already have it, install TinyTeX through Quarto:

```bash
quarto install tinytex
```

Quarto generates both formats: LaTeX is required for PDF, but not DOCX. Run the following commands from the repository root.

#### Build everything and run all checks

Before publishing changes:

```bash
python scripts/build_materials.py --full
```

This validates the materials, runs both test files, executes all 18 notebooks locally and through simulated Colab setup, then generates the PDF/DOCX handouts and solutions for HO02–HO10. A failed check stops the build. The Colab simulation runs locally, not on the hosted service.

#### Rebuild all handouts or just one

| Task | Command |
|---|---|
| All handouts and solutions, with quick validation | `python scripts/build_materials.py` |
| Only the HOXX handout, in PDF and DOCX | `quarto render hands-on/hoXX/handout.qmd --to all` |
| Only the HOXX worked solution, in PDF and DOCX | `quarto render hands-on/hoXX/solutions/handout.qmd --to all` |

Replace `hoXX` with the desired Hands-On folder. Direct `quarto render` commands render only, and do not run the repository checks. Use `python scripts/validate_materials.py` for quick validation when rebuilding individual files.

Outputs stay beside their QMD source: for example, `hands-on/hoXX/handout.qmd` produces `hands-on/hoXX/handout.pdf` and `hands-on/hoXX/handout.docx`. Inspect changed layouts and include the regenerated files with your source changes.

### Check changes without rebuilding

| When | Command |
|---|---|
| Shared code, dependencies, or data changed | `python scripts/validate_materials.py --full` |
| One notebook pair changed, for example, HOXX | `python scripts/check_notebooks.py --ho XX` |
