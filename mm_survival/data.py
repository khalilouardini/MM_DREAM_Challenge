import pandas as pd
import numpy as np
import logging

def find_common_patients(df_clin, df_counts_tpm, df_counts):
    """prepares gene expression data for training.
    Parameters
    ----------
    df_clin : pandas.DataFrame
        Data-frame containing the clinical data
    df_tpm_counts: pandas.DataFrame
        Data-frame containing the TPM normalized RNA-seq data
    df_counts: pandas.DataFrame
        Data-frame containing the raw counts RNA-seq data
    Returns
    -------
    List[str]
        intersection of common patients
    """
    clinical_patients = list(df_clin['Patient'].values)
    seq_patients = list(df_counts_tpm.drop(['Entrez_ID'], axis=1).columns)
    seq_patients2 = list(df_counts.drop(['GENE_ID'], axis=1).columns)

    # Find intersection
    common_patients = list(set(seq_patients) & set(clinical_patients) & set(seq_patients2))
    return common_patients

def select_genes(df, entrez_id):
    """prepares gene expression data for training.
    Parameters
    ----------
    df : pandas.DataFrame
        Data-frame containing TPM normalized gene expression data
    entrez_id: List[int]
        List with Entrez_ID of the subset of genes to extract.
    Returns
    -------
    pandas.DataFrame
        Data-frame with selected genes
    """

    logging.info("Preparing gene expression data")

    df_train = df.loc[df['Entrez_ID'].isin(entrez_id), :]
    # Mapping each gene to a unique index
    idx_to_gene = {i: int(df.iloc[i]['Entrez_ID']) for i in range(df_train.shape[0])}

    df_train = df_train.drop(['Entrez_ID'], axis=1)
    df_train = np.log(1 + df_train)
    
    return df_train, idx_to_gene

def encode_binary(df, list_features):
    """One hot encoding of binary features.
    Parameters
    ----------
    df : pandas.DataFrame
        Data-frame binary features
    list features : List[str] 
        list of binary features to transform
    Returns
    -------
    pandas.DataFrame
        processed Data-frame
    """
    for feat in list_features:
        unique = df[feat].unique()
        replace_dict = {unique[0]: 1, unique[1]: 0}
        df.loc[:, feat] = df.loc[:, feat].replace(replace_dict)
    return df

def binning_feature(df, feat, bins):
    """One hot encoding of binary features.
    Parameters
    ----------
    df : pandas.DataFrame
        Data-frame binary features
    list features : List[str] 
        list of binary features to transform
    Returns
    -------
    pandas.DataFrame
        processed Data-frame
    """
    labels_bins = [1, 2, 3, 4, 5, 6]

    df["D_Age"] = pd.cut(df[feat],
                            bins=bins,
                            labels=labels_bins
                            )
    return df

