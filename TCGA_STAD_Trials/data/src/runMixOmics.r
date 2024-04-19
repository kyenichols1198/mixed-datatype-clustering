# Kye D Nichols
# Script runs MixOmics and produces output

#citation('mixOmics')

#Kim-Anh Le Cao, Florian Rohart, Ignacio Gonzalez, Sebastien Dejean with key contributors Benoit Gautier, Francois
# Bartolo, contributions from Pierre Monget, Jeff Coquery, FangZou Yao and Benoit Liquet. (2016). mixOmics: Omics
# Data Integration Project. R package version 6.1.1. https://CRAN.R-project.org/package=mixOmics

#!/usr/bin/env Rscript
library(mixOmics)
citation('mixOmics')
library(readr)
#set.seed(123)

mirna_tag = "_miRNAseq.csv"
mrna_tag = "_RNAseq.csv"
methyl_tag = "_methyl.csv"
protein_tag = "_Protein.csv"
label_tag = "_labels.csv"
tag ="TCGA-STAD-GI.subtype.noprot"

wd <- getwd()
setwd("..")
parent <- getwd()
setwd(path(wd, "data_proc"))

list.keepX = c(20, 20)
list.keepY = c(20, 20)
feature_params = c(5:9, seq(10, 18, 2), seq(20,30,5))
folds1 = 6
folds2 = 3
ncomp = 5
nrepeat1 = 10
nrepeat2 = 1


args = commandArgs(trailingOnly=TRUE)
if (length(args)!=1) {
  stop("Enter one argument corresponding to file tags", call.=FALSE)
} else {
  tag = args[0]
}

all_files <- list.files(getwd())
select_files <- c()
for (file in all_files) {
    if grepl(tag, file) {
        select_files <- c(select_files, file)
    }
}

data <- list()
test.keepX <- list()
counter = 0
for (file_path in file_paths) {
    print(file_path)
    if (file.exists(file_path)) {
        file_tag = paste0("Ambig", counter)
        if (endsWith(file_path, mirna_tag)){
            data$miRNA <- data.matrix(read.delim(file_path, sep=","))
            test.keepX$miRNA <- feature_params
        }
        if (endsWith(file_path, mrna_tag)) {
            data$mRNA <- data.matrix(read.delim(file_path, sep=","))
            test.keepX$mRNA = feature_params
        }
        if (endsWith(file_path, methyl_tag)) {
            data$Methyl <- data.matrix(read.delim(file_path, sep=","))
            test.keepX$Methyl <- feature_params
        }
        if (endsWith(file_path, protein_tag)) {
            data$Protein <- data.matrix(read.delim(file_path, sep=","))
            test.keepX$Protein <- feature_params
        }
    } else {
      print(paste0(file_path, " does not exist."))
    }
}

#lapply(data, dim)
print(test.keepX)
print(data)
lbls_df = read.delim(label_path, sep=",")
Y = lbls_df$labels
design = matrix(0.1, ncol = length(data), nrow = length(data), dimnames = list(names(data), names(data)))
diag(design) = 0

basic.diablo.model = block.splsda(X = data, Y = Y, ncomp = ncomp, design = design, near.zero.var=TRUE)
perf.diablo = perf(basic.diablo.model, validation = 'Mfold', folds = folds1, nrepeat = nrepeat1, near.zero.var = TRUE)

performance_path = path(paste0(tag, "_mixomics_perf"), ext = ".png")
png(performance_path)
plot(perf.diablo)
dev.off()

BPPARAM <- BiocParallel::MulticoreParam(workers = parallel::detectCores()-1, progressBar=TRUE)
# Windows: BPPARAM <- SnowParallel::MulticoreParam(workers = parallel::detectCores()-1)
ncomp = perf.diablo$choice.ncomp$WeightedVote["Overall.BER", "centroids.dist"]
#perf.diablo$choice.ncomp$WeightedVote
tune.TCGA = tune.block.splsda(X = data,
                              Y = Y,
                              ncomp = ncomp,
                              test.keepX = test.keepX,
                              design = design,
                              validation = 'Mfold',
                              folds = folds2,
                              nrepeat = nrepeat2,
                              dist = "centroids.dist",
                              BPPARAM=BPPARAM,
                              near.zero.var = TRUE
                            )

tune_model_rds_path = path(paste0(tag, "_tune_model"), ext = ".rds")
saveRDS(tune.TCGA, tune_model_rds_path)


list.keepX = tune.TCGA$choice.keepX
final.diablo.model = block.splsda(X = data,
                                  Y = Y,
                                  ncomp = ncomp,
                                  keepX = list.keepX,
                                  design = design,
                                  near.zero.var = TRUE)
model_rds_path = path(paste0(tag, "MixOmicsModel"), ext = ".rds")
saveRDS(final.diablo.model, model_rds_path)

png(path(paste0(tag, "_plotDiablo"), ext = ".png"))
plotDiablo(final.diablo.model, ncomp = 1)
dev.off()

png(path(paste0(tag, "_plotIndiv"), ext = ".png"))
plotIndiv(final.diablo.model, ind.names = FALSE, legend = TRUE, title = 'DIABLO Sample Plots')
dev.off()


png(path(paste0(tag, "_plotArrow"), ext = ".png"))
plotArrow(final.diablo.model, ind.names = FALSE, legend = TRUE, title = 'DIABLO')
dev.off()


png(path(paste0(tag, "_plotVar"), ext = ".png"))
plotVar(final.diablo.model, var.names = FALSE,
        style = 'graphics', legend = TRUE,
        pch = c(16, 17, 15), cex = c(2,2,2),
        col = c('darkorchid', 'brown1', 'lightgreen'))
dev.off()


write.csv(final.diablo.model$X$variates$mRNA, path(paste0(tag, "_MixOmicsModel_RNAseq"), ext = ".csv"))
write.csv(final.diablo.model$X$variates$miRNA, path(paste0(tag, "_MixOmicsModel_miRNAseq"), ext = ".csv"))
write.csv(final.diablo.model$X$variates$Methyl, path(paste0(tag, "_MixOmicsModel_methyl"), ext = ".csv"))
#write.csv(final.diablo.model$X$variates$Protein, path(paste0(tag, "_MixOmicsModel_Protein"), ext = ".csv"))
write.csv(final.diablo.model$Y, path(paste0(tag, "_MixOmicsModel_labels"), ext = ".csv"))

#write.csv(df$variates$mRNA, paste0(tag, "_MixOmicsModel_RNAseq.csv"))
#write.csv(df$variates$miRNA, paste0(tag, "_MixOmicsModel_miRNAseq.csv"))
#write.csv(df$variates$Methyl, paste0(tag, "_MixOmicsModel_methyl.csv"))
#write.csv(data.frame(df$Y), paste0(tag, "_MixOmicsModel_labels.csv"))
