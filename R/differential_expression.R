---
title: "Differential Analysis"
output: html_notebook
---

#install.packages("BiocManager")
#BiocManager::install('limma')
#BiocManager::install('ggplot2')
#BiocManager::install("DESeq2")
#install.packages("htmltools")
#install.packages("rmarkdown")


library(rmarkdown)
library(htmltools)



library( "DESeq2" )
#browseVignettes("DESeq2")

library(ggplot2)


# Import CountData
counts_path <- "/Users/khalilouardini/Desktop/Job Search/Owkin/MM_DREAM_Challenge/exploration/data/gene_expression/differential_expression/count_gene_expression.csv"
countData <- read.csv(counts_path, header = TRUE, sep = ",")
head(countData)

# Import clinical metadata
