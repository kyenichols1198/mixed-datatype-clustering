#!/usr/bin/env Rscript

# Author: Kye D Nichols
# This script runs MixOmics reductions
# Documentation of multi-omics method:
# BRCA case study: http://mixomics.org/mixdiablo/diablo-tcga-case-study/


library(mixOmics)
citation('mixOmics')

#set.seed(123)

###################### Hyperparameters ################
list.keepX = c(20, 20) # number eatures to keep for each block initially
list.keepY = c(20, 20) # number eatures to keep for each block initially
feature_params = c(5:9, seq(10, 18, 2), seq(20,30,5)) # number of folds when tuning number of clusters
folds1 = 6  # number of folds when tuning number of clusters
folds2 = 3 # number of folds when tuning number of features
nrepeat1 = 10 # number of clusters
nrepeat2 = 1 # number of repeats for tuning number of features
near_zero = TRUE # near zero variance for some features
design_scalar = 0.1 # low value prioritise the discriminative ability of the model
dynamic_ncomp = TRUE # tune number of components or keep fixed
unsupervised = FALSE # run unsupervised sPLSA
#######################################################

# detect operating system
# adapted from: https://www.r-bloggers.com/2015/06/identifying-the-os-from-r/
get_os <- function(){
  sysinf <- Sys.info()
  if (!is.null(sysinf)){
      operating_sys <- sysinf['sysname']
    if (operating_sys == 'Darwin')
        operating_sys <- "osx"
  } else {
    os <- .Platform$OS.type
    if (grepl("^darwin", R.version$os))
        operating_sys <- "osx"
    if (grepl("linux-gnu", R.version$os))
        operating_sys <- "linux"
  }
  tolower(operating_sys)
}

# setup number of jobs using BBPARAM multi-core
# path seperator is diff on windows
dir_sep <- "/"
myos <- "linux"
num_jobs <-as.integer(parallelly::availableCores()-2)
if (get_os() == "linux" || get_os() == "osx") {
    print("Found Linux or OSX")
    BPPARAM <- BiocParallel::MulticoreParam(workers = num_jobs) # Mac
} else {
    print("Found Windows")
    BPPARAM <- SnowParallel::MulticoreParam(workers = num_jobs) # Windows
    dir_sep <- "\\"
}

# get command line arguments:
#  tag is a unique string correspoding to output names
#  mydir is the input directory
#  outdir is the output directory
#  results directory is where to put the plots
#  ncomp is the number of components to use
if (length(commandArgs(trailingOnly = TRUE)) > 0) {
  tag <- commandArgs(trailingOnly = TRUE)[1]
  mydir <- commandArgs(trailingOnly = TRUE)[2]
  outdir <- commandArgs(trailingOnly = TRUE)[3]
  resultsdir <- commandArgs(trailingOnly = TRUE)[4]
  ncomp <- as.integer(commandArgs(trailingOnly = TRUE)[5])

  # output file names: composed is n components, the first is decomposed by omics type
  output_path <- paste0(outdir,dir_sep, tag,  "_MixOmics_Embedding.csv")
  label_path <- paste0(outdir,dir_sep, tag, "_MixOmics_Embedding_Composed.csv")
  
  # these are the strings added by processing file and recognized here
  mirna_tag = "_miRNAseq.csv"
  mrna_tag = "_RNAseq.csv"
  methyl_tag = "_methyl.csv"
  protein_tag = "_Protein.csv"
  label_tag = "_labels.csv"
  lbls_df = data.frame()

  # get all the files in the input directory
  wd <- getwd()
  if (!file.exists(output_path)) {
      print("Generating MixOmics Embeddings")
      input_dir <-paste0(getwd(), dir_sep, mydir)
      all_files <- list.files(input_dir)
      select_files <- c()
      for (file in all_files) {
          if (grepl(tag, file)) {
              select_files <- c(select_files, file)
          }
      }
      data <- list()
      test.keepX <- list()
      counter = 0
      # get all datasets and add feature parameters
      setwd(input_dir)
      for (file_path in select_files) {
          if (file.exists(file_path)) {
              if (endsWith(file_path, mirna_tag)){
                  data$miRNA <- data.matrix(read.delim(file_path, sep=",", row.names="X"))
                  test.keepX$miRNA <- feature_params
              }
              if (endsWith(file_path, mrna_tag)) {
                  data$mRNA <- data.matrix(read.delim(file_path, sep=",", row.names="X"))
                  test.keepX$mRNA = feature_params
              }
              if (endsWith(file_path, methyl_tag)) {
                  data$Methyl <- data.matrix(read.delim(file_path, sep=",", row.names="X"))
                  test.keepX$Methyl <- feature_params
              }
              if (endsWith(file_path, label_tag)) {
                  lbls_df <- read.delim(file_path, sep=",", row.names="X")
              }
          } else {
              print(paste0(file_path, " does not exist."))
          }
      }
      
      # create design matrix
      setwd(dirname(input_dir))
      Y = lbls_df$labels
      design = matrix(design_scalar, ncol = length(data), nrow = length(data), dimnames = list(names(data), names(data)))
      diag(design) = 0

      # tune the number of components if necessary
      if (dynamic_ncomp) {
              # if sPLS either supervised or unsupervised...
              if (!(unsupervised)) {
                  basic.diablo.model = block.splsda(X = data, Y = Y, ncomp = ncomp, design = design, near.zero.var=near_zero)
                  perf.diablo = perf(basic.diablo.model, validation = 'Mfold', folds = folds1, nrepeat = nrepeat1, near.zero.var = near_zero)
                  tag=paste0(tag, "_splsa_supervised")
                  performance_path = paste0(resultsdir, dir_sep, tag, "_mixomics_perf.png")
                  png(performance_path)
                  plot(perf.diablo)
                  dev.off()
                  ncomp = perf.diablo$choice.ncomp$WeightedVote["Overall.BER", "centroids.dist"]
              } else {
                  # wrapper.rgcca is purely unsupervised - but doesn't provide images or tuning
                  final.diablo.model = wrapper.rgcca(X = data, ncomp = ncomp, design = design, near.zero.var=near_zero)
                  tag=paste0(tag, "_splsa_unsupervised")
              }
            
       }
        if (!(unsupervised)) {
            # if supervised tune the number of features
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
                                        near.zero.var = near_zero
                                        )
          # save the tuned model as an RDS
          tune_model_rds_path = paste0(resultsdir, dir_sep, tag,  "_tune_model.rds")
          saveRDS(tune.TCGA, tune_model_rds_path)
          
          # rerun with optimal parameters
          list.keepX = tune.TCGA$choice.keepX
          final.diablo.model = block.splsda(X = data,
                                            Y = Y,
                                            ncomp = ncomp,
                                            keepX = list.keepX,
                                            design = design,
                                            near.zero.var = near_zero
                                            )
                                            
           #png(paste0(resultsdir, dir_sep, tag, "_plotDiablo.png"))
           #plotDiablo(final.diablo.model, ncomp = 1)
           #dev.off()
           
           #save the results and plots for supervised method
           png(paste0(resultsdir, dir_sep, tag, "_plotIndiv.png"))
           plotIndiv(final.diablo.model, ind.names = FALSE, legend = TRUE, title = 'DIABLO Sample Plots')
           dev.off()
           
           png(paste0(resultsdir, dir_sep, tag, "_plotArrow.png"))
           plotArrow(final.diablo.model, ind.names = FALSE, legend = TRUE, title = 'DIABLO')
           dev.off()
                                            
           png(paste0(resultsdir, dir_sep, tag, "_plotVar.png"))
           plotVar(final.diablo.model,
                   var.names = FALSE,
                   style = 'graphics', legend = TRUE,
                   pch = c(16, 17, 15), cex = c(2,2,2),
                   col = c('darkorchid', 'brown1', 'lightgreen')
                   )
           dev.off()
      }

      model_rds_path = paste0(resultsdir, dir_sep, tag, "_MixOmicsModel.rds")
      saveRDS(final.diablo.model, model_rds_path)
      
      # save the variates which shows the component values for clusterng
      rna_df_out <- final.diablo.model$variates$mRNA
      colnames(rna_df_out) <- paste0(colnames(rna_df_out), '-rna')
      mirna_df_out <- final.diablo.model$variates$miRNA
      colnames(mirna_df_out) <- paste0(colnames(mirna_df_out), '-mirna')
      methyl_df_out <- final.diablo.model$variates$Methyl
      colnames(methyl_df_out) <- paste0(colnames(methyl_df_out), '-rna')
      merged_df <- cbind(rna_df_out, mirna_df_out, methyl_df_out)
      
      rownames(merged_df) <- rownames(lbls_df)
      write.csv(merged_df, output_path)
      # un-decomposed variates can be used for clustering
      write.csv(final.diablo.model$variates$Y, label_path)

      # save the loadings containing feature contribution to components
      rna_df_load <- final.diablo.model$loadings$mRNA
      write.csv(rna_df_load, paste0(resultsdir, dir_sep, tag,  "_MixOmics_mRNA_Loadings.csv"))

      mirna_df_load <- final.diablo.model$loadings$miRNA
      write.csv(mirna_df_load, paste0(resultsdir, dir_sep, tag,  "_MixOmics_miRNA_Loadings.csv"))

      methyl_df_load <- final.diablo.model$loadings$Methyl
      write.csv(methyl_df_load, paste0(resultsdir, dir_sep, tag,  "_MixOmics_Methyl_Loadings.csv"))
      
      # if unsupervised show component contrib accross labels
      if (!(unsupervised)) {
          y_df_load <- final.diablo.model$loadings$Y
          write.csv(methyl_df_load, paste0(resultsdir, dir_sep, tag,  "_MixOmics_Y_Loadings.csv"))
      }

  } else {
      print("Embeddings Already Exists")
  }
} else {
  print("No arguments provided")
}

