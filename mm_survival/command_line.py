import sys
import logging
import click
from mm_survival import pipelines

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
              prompt='Path to the TPM normalized gene expression data',
              help='Path to the TPM normalized gene expression data')
@click.option('--count_rna_filename',
              type=click.Path(exists=True),
              default= 'exploration/data/gene_expression/MMRF_CoMMpass_IA9_E74GTF_Salmon_Gene_Counts.txt',
              prompt='Path to the raw counts gene expression data',
              help='Path to the raw counts gene expression data')
@click.option('--clinical_filename',
              type=click.Path(exists=True),
              default='exploration/data/clinical/sc3_Training_ClinAnnotations.csv',
              prompt='Path to the data directory',
              help='Path to the data directory')
@click.option('--DE_genes_filename',
              type=click.Path(exists=True),
              default='exploration/data/gene_expression/differential_expression/DE_genes.txt',
              prompt='Path to the list of differentially expressed genes',
              help='List of differentially expressed genes')
@click.option('--signature_gene_filename',
              default='exploration/data/gene_expression/differential_expression/signature_genes.txt',
              prompt='Path to the list of signature genes',
              help='List of signature genes'
              )
@click.option('--ensembling',
              default=True,
              prompt='Whether to use model ensembling',
              help='Want to use ensembling?'
              )
@click.option('--n_estimators',
              default=500,
              prompt='Number of estimators in our ensembling model',
              help='Number of estimators in our ensembling model'
              )          
@click.option('--do_hyperopt',
              prompt='Whether to run hyperparameter tuning (random search)',
              help='Whether to run hyperparameter tuning (random search)'
              )
@click.option('--run_inference',
              prompt='Whether to run inference on test set',
              help='Whether to run inference on test set'
              )

def mm_survival_analysis(tpm_rna_filename, count_rna_file, clinical_file,
                    DE_genes_filename, signature_gene_filename, model, n_estimators, do_hyperopt, run_inference):
    if run_inference:
        pipelines.run_inference(tpm_rna_filename, count_rna_file, clinical_file,
                               DE_genes_filename, signature_gene_filename, model, n_estimators)
    else:
        pipelines.run_survival_analysis(tpm_rna_filename, count_rna_file, clinical_file,
                               DE_genes_filename, signature_gene_filename, model, n_estimators, do_hyperopt)