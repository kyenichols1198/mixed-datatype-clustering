library(TCGAbiolinks)
library(SummarizedExperiment)
library(DT)

convert_string_to_list <- function(input_string) {
  parts <- strsplit(input_string, "-")[[1]]
  result <- paste0("TCGA-", toupper(parts))
  return(result)
}

mystr = "STAD" # set default
if (length(commandArgs(trailingOnly = TRUE)) > 0) {
  mystr <- commandArgs(trailingOnly = TRUE)[1]
  mydir <- commandArgs(trailingOnly = TRUE)[2]
} else {
  print("No argument provided")
}

wkdir <- getwd()
tcga_str <- paste0("TCGA-", mystr)
if (grepl("-", mystr)) {
    tcga_arr <- convert_string_to_list(mystr)
 } else {
    tcga_arr <- tcga_str
 }

dir.create(file.path(wkdir, mydir, tcga_str), showWarnings = FALSE)
subdir_path <- file.path(wkdir, mydir, tcga_str)
setwd(subdir_path)

rna_path = paste0(tcga_str,"-RNASeq.rds")
if (file.exists(rna_path)) {
    print("RNA data is downloaded...")
} else {
    query_rna = GDCquery(
                project = tcga_arr,
                data.category = "Transcriptome Profiling",
                data.type = "Gene Expression Quantification",
                workflow.type = "STAR - Counts"
            )
    rnaseq_res = getResults(query_rna)
    datapath = file.path(tempdir(), "GDCdata")
    GDCdownload(query_rna, directory = datapath)
    stad_se = GDCprepare(query_rna, directory = datapath)
    saveRDS(stad_se, rna_path)
    
    df <- readRDS(rna_path)
    log2_TPM <- data.frame(assay(df,4))
    write.csv(log2_TPM, paste0(tcga_str, "-Counts_tpm.csv"))
}

methyl_path <- paste0(tcga_str, "-Methyl450.rds")
if (file.exists(methyl_path)) {
    print("Methylation data is downloaded...")
} else {
    query_met.hg38 <- GDCquery(
        project= tcga_arr,
        data.category = "DNA Methylation",
        data.type = "Methylation Beta Value",
        platform = "Illumina Human Methylation 450"
    )
    GDCdownload(query_met.hg38)
    data.hg38 <- GDCprepare(query_met.hg38)
    saveRDS(data.hg38, methyl_path)
    
    df <- readRDS(methyl_path)
    beta <- data.frame(assays(df,1, withDimnames=TRUE))
    write.csv(beta, paste0(tcga_str, "-Methyl450-Beta.csv"))
}

mirna_path <-paste0(tcga_str, "-miRNA.rds")
if (file.exists(mirna_path)) {
    print("miRNA data is downloaded...")
} else {
    query.mirna <- GDCquery(
        project = tcga_arr,
        experimental.strategy = "miRNA-Seq",
        data.category = "Transcriptome Profiling",
        data.type = "miRNA Expression Quantification"
    )
    GDCdownload(query.mirna)
    mirna <- GDCprepare(query.mirna)
    saveRDS(mirna,mirna_path)
    df <- readRDS(mirna_path)
    write.csv(df, paste0(tcga_str, "-miRNA.csv"))
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
subtype_fname= "PANCAN_Subtype.csv"
get_subtypes(subtype_fname)

# has slightly different format and clinical data is sparse
download_clinical <- function(stype, tcga_arr) {
    clinical_file_path <- paste0(stype, "_clinical.csv")
    if (file.exists(clinical_file_path)) {
        clinical <- read.csv(clinical_file_path)
    } else {
        clinical <- GDCquery_clinic(project = tcga_arr, type = "clinical")
        write.csv(clinical, clinical_file_path)
    }
    clinical
}
download_clinical(mystr, tcga_arr)
