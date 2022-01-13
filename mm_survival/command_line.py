import sys
import logging
import click
from mm_survival import pipelines
import numpy as np

logging.basicConfig(
    format='[%(asctime)s|%(module)s.py|%(levelname)s]  %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO,
    stream=sys.stdout
)

@click.command()
@click.option('--tpm_rna_filename',
              type=click.Path(exists=True),
              default='exploration/data/gene_expression/MMRF_CoMMpass_IA9_E74GTF_Salmon_entrezID_TPM_hg19.csv',
              prompt='Path to the TPM normalized gene expression data (TRAIN)',
              help='Path to the TPM normalized gene expression data (TRAIN)')
@click.option('--tpm_rna_filename_test',
              type=click.Path(exists=True),
              default='exploration/data/gene_expression/MMRF_CoMMpass_IA9_E74GTF_Salmon_entrezID_TPM_hg19.csv',
              prompt='Path to the TPM normalized gene expression data (TEST)',
              help='Path to the TPM normalized gene expression data (TEST)')
@click.option('--count_rna_filename',
              type=click.Path(exists=True),
              default= 'exploration/data/gene_expression/MMRF_CoMMpass_IA9_E74GTF_Salmon_Gene_Counts.txt',
              prompt='Path to the raw counts gene expression data (TRAIN)',
              help='Path to the raw counts gene expression data (TRAIN)')
@click.option('--clinical_filename',
              type=click.Path(exists=True),
              default='exploration/data/clinical/sc3_Training_ClinAnnotations.csv',
              prompt='Path to the clinical data (TRAIN)',
              help='Path to the clinical data (TRAIN)')
@click.option('--clinical_filename_test',
              type=click.Path(exists=True),
              default='exploration/data/clinical/sc3_Training_ClinAnnotations.csv',
              prompt='Path to the clinical data (TEST)',
              help='Path to the clinical data (TEST)')
@click.option('--de_genes_filename',
              type=click.Path(exists=True),
              default='exploration/data/gene_expression/differential_expression/DE_genes.txt',
              prompt='Path to the list of differentially expressed genes',
              help='List of differentially expressed genes')
@click.option('--signature_genes_filename',
              default='exploration/data/gene_expression/differential_expression/signature_genes.txt',
              prompt='Path to the list of signature genes',
              help='List of signature genes'
              )
@click.option('--only_clinical',
              default=False,
              prompt='Whether to use the clinical data only',
              help='Whether to use the clinical data only?'
              )
@click.option('--model',
              default='RF',
              prompt='The model to train among [RF, XG, logreg]',
              help='Which model do you want to train?'
              )
@click.option('--n_estimators',
              default=200,
              prompt='Number of estimators in our ensembling model',
              help='Number of estimators in our ensembling model'
              )          
@click.option('--run_inference',
              default=False,
              prompt='Whether to run inference on test set',
              help='Whether to run inference on test set'
              )
@click.option('--survival_analysis',
              default=False,
              prompt='Whether to use a survival analysis model (Cox regression)',
              help='Whether to use a survival analysis model (Cox regression)'
              )
@click.option('--ensembling',
              default=False,
              prompt='Whether to use model ensembling',
              help='Want to use ensembling?'
              )
@click.option('--n_runs',
              default=10,
              prompt='Number of total runs',
              help='Number of total runs'
              )
@click.option('--include_censored',
              default=False,
              prompt='Whether to include the censored patients',
              help='Whether to include the censored patients'
              )

def mm_survival_analysis(tpm_rna_filename, tpm_rna_filename_test, count_rna_filename, clinical_filename,
                    clinical_filename_test, de_genes_filename, signature_genes_filename, model,
                     n_estimators, only_clinical, survival_analysis, ensembling,
                      n_runs, run_inference, include_censored):
    
    final_acc = []
    final_auc = []
    final_precision = []
    final_recall = []
    for i in range(n_runs):
        clf, accs, aucs, precisions, recalls = pipelines.run_pipeline(tpm_rna_filename, count_rna_filename,
                     clinical_filename=clinical_filename,
                     de_genes_filename=de_genes_filename, signature_genes_filename=signature_genes_filename, model=model,
                     n_estimators=n_estimators, only_clinical=only_clinical, survival_analysis=survival_analysis,                    
                     ensembling=ensembling, include_censored=include_censored)
        final_acc += accs
        final_auc += aucs
        final_precision += precisions
        final_recall += recalls

    print("")
    print(" ================ CV Performance average over 10 runs ================ ")
    print("Accuracy : mean: {} | std: {}".format(np.mean(final_acc), np.std(final_acc)))
    print("AUC : mean: {} | std: {}".format(np.mean(final_auc), np.std(final_auc)))
    print("Precision : mean: {} | std: {}".format(np.mean(final_precision), np.std(final_precision)))
    print("Recall : mean: {} | std: {}".format(np.mean(final_recall), np.std(final_recall)))

    if run_inference:
        pipelines.run_inference(clf, tpm_rna_filename_test, clinical_filename_test, de_genes_filename, survival_analysis)