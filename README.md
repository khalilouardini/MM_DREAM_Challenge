# MM DREAM Challenge

![Alt text](mm_image.png?raw=true "MM")

## 0. Description

A Python library for classification of high risk patients with Multiple Myeloma (MM)

For more information, please refer to the **report**.

## 1. Setup
**System requirements**: `Python3`

to install the package, follow these instructions

-   Create a new envrionement with `conda` or `virtualenv`: `conda create -n name_env python=3.7.12`
-   Install the package into your envrionement - `pip install -e 'git+https://github.com/khalilouardini/MM_DREAM_Challenge.git#egg=mm_survival'`. The package will be installed in a `src/` directory.
-   Go to the main deirectory `cd src/mm-survival`.
-   Install requirements - `pip install -r requirements.txt`

## 2. Instructions 

To run the code, you can directly select options from the command line. The options are:
-   `--tpm_rna_filename`: Path to the TPM normalized gene expression data (TRAIN)
-   `--tpm_rna_filename_test`: Path to the TPM normalized gene expression data (TEST)
-   `--count_rna_filename`: Path to the raw counts gene expression data (only TRAIN)
-   `--clinical_filename`: Path to the clinical data (TRAIN)
-   `--clinical_filename_test`: Path to the clinical data (TEST)
-   `--de_genes_filename`: Path to the list of differentially expressed genes 
-   `--signature_genes_filename`: Path to the list of signature genes
-   `--only_clinical`: Whether to use the clinical data only
-   `--model`: The model to train among [RF=Random Forest, XG=XG Boost, logreg=Logistic Regression]
-   `--n_estimators`: Number of estimators (for RF and XG)
-   `--run_inference`: Whether to run inference on test set
-   `--survival_analysis`: Whether to use a survival analysis model (Penalized Cox regression)
-   `--ensembling`: Whether to use model ensembling
-   `--n_runs`: Total number of cross-validation runs
-   `--include_censored`: Whether to include the censored patients

### Reproducing experiments

-  The list of genes entrezID used in the experiments is stored in `exploration/data/gene_expression/differential_expression`.

- [For help, run: `mm_survival_analysis --help` in the commande line]]

- Make sure you to replace `path/to/your/data` with your path, all the other options will be set to default. To reproduce the experiment with the **Random Forest** model, run:

    `mm_survival_analysis --tpm_rna_filename path/to/your/data --count_rna_filename path/to/your/data. --clinical_filename path/to/your/data`

- To run survival analysis with a **Penalized Cox proportional hazard regression** model, run the same command but set `survival_analysis` to True.
    `mm_survival_analysis --tpm_rna_filename path/to/your/data --count_rna_filename path/to/your/data --clinical_filename path/to/your/data --survival_analysis True`

- For interactivre use, you can also simply run `mm_survival_analysis` in your terminal window. the options will be prompted one by one.

### Running predictions on validation data
-   First make sure you have a directory with all the data - train and test. The test data is assumed to have the exact same format as the train data (i.e a csv with the same column names). Run the command line, and add the path of the train and test files:

    `mm_survival_analysis --tpm_rna_filename path/to/your/data --tpm_rna_filename_test path/to/your/data --count_rna_filename path/to/your/data --clinical_filename path/to/your/data --clinical_filename_test path/to/your/data --run_inference True `

-   Or, run `mm_survival_analysis` for interactive use (don't forget to set `--run_inference` to True)

-   If the csv containing the test clinical data has a 'HR_FLAG' column, the evaluation metrics for the test set will be returned. Otherwise, the predictions will be saved in a `results` folder.

## 3 Code organization

-   The exploratory analysis is done on jupyter notebooks in the `exploration/` folder
-   `data.py`: each function in this script is a step of the pre-processing pipeline
-   `models.py`: contains code for fitting classic ML models (with or without hyperparameter search) and reporting evaluation metrics.``
-   `models_survival.py`: contains code for fitting survival models  (Penalized Cox proportional hazard regression) and reporting evaluation metrics.``
-   `pipelines.py`: contains the whole pipeline from pre-processing, to model fitting to inference on unseen data.

**Differential expression (DE)** is implemented separately with DESeq2 in R. All the code related to differential analysis is in the `R/` folder. After running DE, we get a fixed set of genes that we store in `exploration/data/gene_expression/differential_expression/DE_genes.txt`
