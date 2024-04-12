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
output_path = "MixOmicsModel.rds"

list.keepX = c(20, 20)
list.keepY = c(20, 20)
feature_params = c(5:9, seq(10, 18, 2), seq(20,30,5))
folds1 = 6
folds2 = 3
ncomp = 5
nrepeat1 = 10
nrepeat2 = 1

file_paths = FALSE
output_path = FALSE
args = commandArgs(trailingOnly=TRUE)
if (length(args)==0) {
  stop("Enter input files", call.=FALSE)
} else if (length(args)<=5 & length(args)>2) {
  all_paths = head(args, -1)
  file_paths = head(args, -1)
  label_path = tail(args, n=1)
} else {
  stop("Wrong Number of arugments: Provide input omics csv, label csv and output fname (3-6 arguments)", call.=FALSE)
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

performance_path = "mixomics_perf.png"
png(performance_path)
plot(perf.diablo)
dev.off()

BPPARAM <- BiocParallel::MulticoreParam(workers = parallel::detectCores()-1)
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
#                              progressBar=TRUE,
                              near.zero.var = TRUE
                            )

tune_model_rds_path = "tune_model.rds"
saveRDS(tune.TCGA, tune_model_rds_path)


list.keepX = tune.TCGA$choice.keepX
final.diablo.model = block.splsda(X = data,
                                  Y = Y,
                                  ncomp = ncomp,
                                  keepX = list.keepX,
                                  design = design,
                                  near.zero.var = TRUE)
saveRDS(final.diablo.model, output_path)

png("plotDiablo.png")
plotDiablo(final.diablo.model, ncomp = 1)
dev.off()

png("plotIndiv.png")
plotIndiv(final.diablo.model, ind.names = FALSE, legend = TRUE, title = 'DIABLO Sample Plots')
dev.off()

png("plotArrow.png")
plotArrow(final.diablo.model, ind.names = FALSE, legend = TRUE, title = 'DIABLO')
dev.off()


png("plotVar.png")
plotVar(final.diablo.model, var.names = FALSE,
        style = 'graphics', legend = TRUE,
        pch = c(16, 17, 15), cex = c(2,2,2),
        col = c('darkorchid', 'brown1', 'lightgreen'))
dev.off()

write.csv(final.diablo.model$X$variates$mRNA, "MixOmicsModel_RNAseq.csv")
write.csv(final.diablo.model$X$variates$miRNA, "MixOmicsModel_miRNAseq.csv")
write.csv(final.diablo.model$X$variates$Methyl, "MixOmicsModel_RNAseq.csv")
write.csv(final.diablo.model$X$variates$Protein, "MixOmicsModel_Protein.csv")


