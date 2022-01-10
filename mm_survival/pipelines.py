import logging
import pandas as pd
import numpy as np
import os
from mm_survival import data, models

def run_survival_analysis(tpm_rna_filename, count_rna_file, clinical_file,
                    DE_genes_filename, signature_gene_filename, model, n_estimators, random_search):
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
    mdoel: str
        Model to train [Random Forest ('RF') or XGBoost ('XG')]
    n_estimators: int
        Number of estimators in ensembling model
    """
    logging.info('Starting the pipeline')

    # Import data
    df_counts_tpm = pd.read_csv(tpm_rna_filename, index_col=False)
    df_counts = pd.read_csv(count_rna_file, delimiter = "\t")
    df_clin = pd.read_csv(clinical_file)

    # Import genes
    with open(DE_genes_filename, 'r') as f:
        lines = f.readlines()
    DE_genes = [int(line.split('\n')[0]) for line in lines]
    with open(signature_gene_filename, 'r') as f:
        lines = f.readlines()
    signature_genes = [int(line.split('\n')[0]) for line in lines]

    print("We have {} genes in the raw counts gene expression matrix".format(df_counts['GENE_ID'].unique().shape[0]))
    print("We have {} genes in the raw counts gene expression matrix".format(df_counts_tpm['Entrez_ID'].unique().shape[0]))

    print("N° of patients in the MMRF cohort, with RNAseq available RNA-seq data: {}".format(df_counts.shape[1]))
    print("N° of patients in the MMRF cohort, with RNAseq available TPM-normalized RNA-seq data: {}".format(df_counts_tpm.shape[1]))

    # Rename Gene ID column
    df_counts_tpm = df_counts_tpm.rename(columns = {'Unnamed: 0':'Entrez_ID'})

    # Standardize patients ID's
    df_clin['Patient'] += '_1_BM'

    # Pre-processing pipeline for train set

    # 1. Finiding the common patients between the 3 data sources, and select them
    common_patients = data.find_common_patients(df_clin, df_counts_tpm, df_counts)
    print("Number of patients with clinical and sequencing data: {}".format(len(common_patients)))
    df_counts = df_counts.loc[:, ['GENE_ID'] + common_patients]
    df_counts_tpm = df_counts_tpm.loc[:, ['Entrez_ID'] + common_patients]
    df_clin = df_clin[df_clin['Patient'].isin(common_patients)]

    # 2. We only keep non-censored patients (!! /!\)
    df_clin = df_clin[df_clin['HR_FLAG'] != "CENSORED"]
    df_counts_tpm = df_counts_tpm.loc[:, ['Entrez_ID'] + list(df_clin['Patient'])]
    df_counts = df_counts.loc[:, ['GENE_ID'] + list(df_clin['Patient'])]

    # 3. Select Differentially expressed genes
    print("Total number of genes in the dataset: {}".format(len(DE_genes)))
    df_train, idx_to_gene = data.select_genes(df_counts_tpm)

    # 4. Merge with clinical data
    df_train["D_ISS"] = df_clin["D_ISS"].values
    df_train["D_Age"] = df_clin["D_Age"].values
    df_train["D_Gender"] = df_clin["D_Gender"].values

    # 5. Prepare clinical features
    ## a. Encode binary variables
    df_train = data.encode_binary(df_train, ["D_Gender"])
    ## b. Binning the "D_Age" variable
    df_train = data.encode_binary(df_train, "D_Age", [0, 35, 50, 60, 70, 80, 200])
    ## c. Missing values (???)
    df_train[['D_ISS']] = df_train[['D_ISS']].fillna(2)

    # Fit model
    #KEEP_FEATURES = [col for col in processed_df.columns if col not in ['price', 'logprice', 'drug_id']]
    #if not random_search:
    #    model, mape_score, mse_score, mae_score = models.fit_cv(processed_df,
    #                                                            KEEP_FEATURES,
    #                                                            model=model,
    #                                                            n_estimators=n_estimators
    #                                                            )
    #else:
    #    logging.info("=== Hyperparameter Tuning ===")
    #    model, mape_score, mse_score, mae_score = models.fit_cv_random_search(processed_df,
    #                                                    KEEP_FEATURES,
    #                                                    model=model
    #                                                    )

    #return model, mape_score, mse_score, mae_score