import logging
import pandas as pd
from mm_survival import data, models
import numpy as np

def run_pipeline(tpm_rna_filename, count_rna_file, clinical_file,
                    DE_genes_filename, signature_gene_filename, model, n_estimators, return_data=True, top_k_genes=40,
                    survival_analysis=False):
    """Data pipeline and predictions on train set.
    Parameters
    ----------
    tpm_rna_filename,: str
        file path of the TPM normalized RNA-seq data 
    tpm_rna_filename: str
        file path of the raw RNA-seq data 
    clinical_file: str
        file path of the TPM normalized RNA-seq data 
    DE_genes: str:
        Informative genes selected by differential analysis
    DE_genes: str:
        Signature genes from the litterature (EMC-92 and UAMS-70)
    model: str
        Model to train [Random Forest ('RF') or XGBoost ('XG')]
    n_estimators: int
        Number of estimators in ensembling model
    """
    logging.info('=== Data pre-processing ===')

    # Import data
    df_tpm = pd.read_csv(tpm_rna_filename, index_col=False)
    df_counts = pd.read_csv(count_rna_file, delimiter = "\t")
    df_clin = pd.read_csv(clinical_file)

    # Rename Gene ID column 
    df_tpm = df_tpm.rename(columns = {'Unnamed: 0':'Entrez_ID'})

    # Import genes
    with open(DE_genes_filename, 'r') as f:
        lines = f.readlines()
    DE_genes = [int(line.split('\n')[0]) for line in lines]
    with open(signature_gene_filename, 'r') as f:
        lines = f.readlines()
    signature_genes = [int(line.split('\n')[0]) for line in lines]

    print("We have {} genes in the raw counts gene expression matrix".format(df_counts['GENE_ID'].unique().shape[0]))
    print("We have {} genes in the TPM normalized gene expression matrix".format(df_tpm['Entrez_ID'].unique().shape[0]))

    print("N° of patients in the MMRF cohort, with RNAseq available RNA-seq data: {}".format(df_counts.shape[1]))
    print("N° of patients in the MMRF cohort, with RNAseq available TPM-normalized RNA-seq data: {}".format(df_tpm.shape[1]))

    # Standardize patients ID's
    df_clin['Patient'] += '_1_BM'

    # Pre-processing pipeline for train set

    # 1. Finiding the common patients between the 3 data sources, and select them
    common_patients = data.find_common_patients(df_clin, df_tpm, df_counts)
    print("Number of patients with clinical and sequencing data: {}".format(len(common_patients)))
    df_counts = df_counts.loc[:, ['GENE_ID'] + common_patients]
    df_tpm = df_tpm.loc[:, ['Entrez_ID'] + common_patients]
    df_clin = df_clin[df_clin['Patient'].isin(common_patients)]


    # 3. Seperate censored and non-censored patients
    ## a. censored
    df_clin_censored = df_clin[df_clin['HR_FLAG'] == "CENSORED"]
    df_tpm_censored = df_tpm.loc[:, ['Entrez_ID'] + list(df_clin_censored['Patient'])]
    ## b. not censored
    df_clin_uncensored = df_clin[df_clin['HR_FLAG'] != "CENSORED"]
    df_tpm_uncensored = df_tpm.loc[:, ['Entrez_ID'] + list(df_clin_uncensored['Patient'])]
    ## c. Raw counts GE data
    #df_counts_censored =  df_counts.loc[:, ['GENE_ID'] + list(df_clin_censored['Patient'])]
    #df_counts = df_counts.loc[:, ['GENE_ID'] + list(df_clin['Patient'])]

    # 4. Select Differentially expressed genes for censored and non-censored patients
    print("Total number of genes in the dataset: {}".format(len(DE_genes)))
    final_genes = DE_genes[:top_k_genes] #+ signature_genes
    df_train, _ = data.select_genes(df_tpm_uncensored, final_genes)
    df_train_censored, _ =  data.select_genes(df_tpm_censored, final_genes)
    df_train_full, __ = data.select_genes(df_tpm, final_genes)

    # 4. Merge with clinical data
    df_train = df_train.T
    df_train["D_ISS"] = df_clin_uncensored["D_ISS"].values
    df_train["D_Age"] = df_clin_uncensored["D_Age"].values

    df_train_censored = df_train_censored.T
    df_train_censored["D_ISS"] = df_clin_censored["D_ISS"].values
    df_train_censored["D_Age"] = df_clin_censored["D_Age"].values

    df_train_full = df_train_full.T
    df_train_full["D_ISS"] = df_clin["D_ISS"].values
    df_train_full["D_Age"] = df_clin["D_Age"].values

    # 5. Prepare clinical features

    ## a. Encode binary variables
    #df_train = data.encode_binary(df_train, ["D_Gender"])

    ## b. Binning the "D_Age" variable
    df_train = data.binning_feature(df_train, "D_Age", [0, 35, 50, 60, 70, 80, 200])
    df_train_censored = data.binning_feature(df_train_censored, "D_Age", [0, 35, 50, 60, 70, 80, 200])
    df_train_full = data.binning_feature(df_train_full, "D_Age", [0, 35, 50, 60, 70, 80, 200])
    ## c. Missing values
    df_train[['D_ISS']] = df_train[['D_ISS']].fillna(2)
    df_train_censored[['D_ISS']] = df_train_censored[['D_ISS']].fillna(2)
    df_train_full[['D_ISS']] = df_train_full[['D_ISS']].fillna(2)

    # 6. Prepare design matrix and labels
    X = df_train.values #.loc[:, ["D_Age", "D_ISS"]].values
    print(X.shape)
    ## a. Classification labels labels
    replace_dict = {'TRUE': 1, 'FALSE': 0}
    y_hr = df_clin['HR_FLAG'].replace(replace_dict).values
    #y_os_clf = df_clin['D_OS_FLAG'].values
    #y_pfs_clf = df_clin['D_PFS_FLAG'].values
    ## b. Regression labels
    y_os = df_clin['D_OS'].values
    y_pfs = df_clin['D_PFS'].values

    # 7. Additional data for regression models
    X_censored = df_train_censored.values
    y_os_censored = df_clin_censored['D_OS'].values
    y_pfs_censored = df_clin_censored['D_PFS'].values

    # 8. 
    inputs = [X, y_hr, y_os, y_pfs]
    inputs_censored = [X_censored, y_os_censored, y_pfs_censored]

    if return_data:
        return df_train, df_train_censored, df_clin_uncensored, df_clin_censored

    # 9. Train model
    logging.info("=== End of Data pre-processing ===")
    clf, accs, aucs = models.fit_cv_clf([X, y_hr], model, n_estimators)
    clf, regressor_os, regressor_pfs, accs, aucs = models.fit_cv_ensemble(inputs, inputs_censored, model, n_estimators)
    logging.info("=== Start model training ===")

    return accs, aucs, X, y_hr

if __name__ == "__main__":
    tpm_rna_filename = 'exploration/data/gene_expression/MMRF_CoMMpass_IA9_E74GTF_Salmon_entrezID_TPM_hg19.csv'
    count_rna_file = 'exploration/data/gene_expression/MMRF_CoMMpass_IA9_E74GTF_Salmon_Gene_Counts.txt'
    clinical_file = 'exploration/data/clinical/sc3_Training_ClinAnnotations.csv'
    DE_genes_filename = 'exploration/data/gene_expression/differential_expression/DE_genes.txt'
    signature_genes_filename = 'exploration/data/gene_expression/differential_expression/signature_genes.txt'


    final_acc = []
    final_auc = []
    for i in range(10):
        accs, aucs = run_pipeline(tpm_rna_filename, count_rna_file, clinical_file,
                    DE_genes_filename, signature_genes_filename, 'RF', 200)
        final_acc.append(np.mean(accs))
        final_auc.append(np.mean(aucs))
    
    print(np.mean(final_acc), np.mean(final_auc))
    print("Done")