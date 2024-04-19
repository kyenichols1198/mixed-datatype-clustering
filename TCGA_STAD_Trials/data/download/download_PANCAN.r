#!/usr/bin/env Rscript
#--------------------------------------------------------------------
# Kye Nichols
# This script downloads all the data needed for KMedoids Paper
# https://xenabrowser.net/datapages/ (Official Xena website)
# https://rdrr.io/github/ShixiangWang/UCSCXenaTools/man/getTCGAdata.html (ShixiangWang)
# Good resource for constructing Xena Queries
# https://shixiangwang.github.io/home/en/tools/ucscxenatools-intro/ (上士闻道 勤而行之)
# https://cran.r-project.org/web/packages/UCSCXenaTools/vignettes/USCSXenaTools.html
library(UCSCXenaTools)
library(TCGAbiolinks)
library(DT)
# install USCSXenaTools:
#      install.packages("UCSCXenaTools")
#      install.packages("remotes")
#      remotes::install_github("ropensci/UCSCXenaTools")
# Cite: Wang et al., (2019). The UCSCXenaTools R package: a toolkit for accessing genomics data
# from UCSC Xena platform, from cancer multi-omics to single-cell RNA-seq.
# Journal of Open Source Software, 4(40), 1627, https://doi.org/10.21105/joss.01627

tcga_subtype_path= "PANCAN_Subtype.csv"
subtypes = c("STAD", "BRCA", "COAD", "CHOL", "LAML")
ds_names_fname = "datasets.txt"


# has slightly different format and clinical data is sparse
download_clinical <- function(stype) {
    clinical_file_path <- paste0(stype, "_clinical.csv")
    if (file.exists(clinical_file_path)) {
        clinical <- read.csv(clinical_file_path)
    } else {
        clinical <- GDCquery_clinic(project = paste0("TCGA-", stype), type = "clinical")
        write.csv(clinical, clinical_file_path)
    }
    clinical
}

# downloads omics data based on dataset name
download_Xena_query <- function(dataset_name) {
    XenaGenerate(subset = XenaDatasets == dataset_name) -> df
    options(use_hiplot = TRUE)
    XenaQuery(df)%>% XenaDownload(destdir = getwd()) -> xe_download
    xe_download
}

# get subtype labels form TCGAbioportal
get_subtypes <- function(subtype_path) {
    if (file.exists(subtype_path)) {
        subtype_data <- read.csv(subtype_path)
    } else {
        subtypes <- PanCancerAtlas_subtypes()
        subtype_data <- DT::datatable(
            data = subtypes,
            filter = 'top',
            options = list(scrollX = TRUE, keys = TRUE, pageLength = 5),
            rownames = FALSE
            )
        subtypes_data = subtype_data[[1]]$data
        write.csv(subtypes_data, subtype_path)
    }
    subtype_data
}

conn <- file(ds_names_fname,open="r")
dss <-readLines(conn)
for (st in subtypes) {download_clinical(st)}
for (ds in dss) { download_Xena_query(ds) }
get_subtypes(tcga_subtype_path)



