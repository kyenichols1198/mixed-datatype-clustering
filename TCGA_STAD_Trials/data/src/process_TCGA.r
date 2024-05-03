if (length(commandArgs(trailingOnly = TRUE)) > 0) {
  inputdir <- commandArgs(trailingOnly = TRUE)[1]
  outputdir <- commandArgs(trailingOnly = TRUE)[2]
} else {
  print("No argument provided")
}

print(paste0("Input directory: ", inputdir))
print(paste0("Output directory: ", outputdir))

