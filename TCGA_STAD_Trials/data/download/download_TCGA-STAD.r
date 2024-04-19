library(TCGAbiolinks)
library(SummarizedExperiment)
# library(UCSCXenaTools)
library(TCGAbiolinks)
library(DT)

query_rna = GDCquery(
            project = 'TCGA-STAD',
            data.category = 'Transcriptome Profiling',
            data.type = 'Gene Expression Quantification',
            workflow.type = 'STAR - Counts'
        )

rnaseq_res = getResults(query_rna)
datapath = file.path(tempdir(), 'GDCdata')
GDCdownload(query_rna, directory = datapath)
stad_se = GDCprepare(query_rna, directory = datapath)
saveRDS(stad_se, "TCGA-STAR-Counts.rds")


query_met.hg38 <- GDCquery(
    project= "TCGA-STAD",
    data.category = "DNA Methylation",
    data.type = "Methylation Beta Value",
    platform = "Illumina Human Methylation 450"
)

GDCdownload(query_met.hg38)
data.hg38 <- GDCprepare(query_met.hg38)
saveRDS(data.hg38, "TCGA-STAD-Methyl450-Beta.rds")

query.rppa <- GDCquery(
    project = "TCGA-STAD",
    data.category = "Proteome Profiling",
    data.type = "Protein Expression Quantification"
)
GDCdownload(query.rppa)
rppa <- GDCprepare(query.rppa)
saveRDS(rppa, "TCGA-STAD-RPPA.rds")

query.mirna <- GDCquery(
    project = "TCGA-STAD",
    experimental.strategy = "miRNA-Seq",
    data.category = "Transcriptome Profiling",
    data.type = "miRNA Expression Quantification"
)

GDCdownload(query.mirna)
mirna <- GDCprepare(query.mirna)
saveRDS(mirna, "TCGA-STAD-miRNA.rds")

#GDCdownload(query.mirna)
#mirna <- GDCprepare(
#    query = query.mirna,
#    save = TRUE,
#    save.filename = "mirna.rda"
#)

df <- readRDS("TCGA-STAR-Counts.rds")
TPM <- data.frame(assay(df,4))
write.csv(TPM, "TCGA-STAR-Counts_tpm.csv")
# "unstranded:1", "stranded_first:2", "stranded_second:3", "tpm_unstrand:4", "fpkm_unstrand:5", "fpkm_uq_unstrand:6"


df <- readRDS("TCGA-STAD-miRNA.rds")
write.csv(df, "TCGA-STAD-miRNA.csv")

df <- readRDS("TCGA-STAD-Methyl450-Beta.rds")
beta <- data.frame(assays(df,1, withDimnames=TRUE))
write.csv(beta, "TCGA-STAD-Methyl450-Beta.csv")


df <- readRDS("TCGA-STAD-RPPA.rds")
write.csv(df, "TCGA-STAD-RPPA.csv")


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
subtype_fname= "PANCAN_Subtype.csv"
get_subtypes(subtype_fname)



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
download_clinical("STAD")
