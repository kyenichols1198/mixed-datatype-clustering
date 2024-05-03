#!/bin/bash

# This script runs the pipeline with...
# MixOmics embedding GI subtype
# Usage bash pipeline.sh -t STAD -name STAD_GI_Subtype -st Subtype_Selected -mode MixOmics -ncomp 5
# CustOmics embedding GI subtype
# Usage bash pipeline.sh -t STAD -name STAD_GI_Subtype -st Subtype_Selected -mode CustOmics -nlatent 20

# MixOmics embedding Immune subtype
# Usage bash pipeline.sh -t STAD -name STAD_Immune_Subtype -st Immune.Subtype -mode MixOmics -ncomp 5
# MixOmics embedding Immune subtype
# Usage bash pipeline.sh -t STAD -name STAD_Immune_Subtype -st Immune.Subtype -mode CustOmics -nlatent 20

# ----------------------------Commonly changed arguments ---------------------------- #
tcga_type="STAD" # can retrieve multiple by adding dash in between
mytag="TCGA-STAD-Test" # tag to add to output file names
label_col_name="Subtype_Selected" # column name of subtypes in PANCANCER subtypes csv
reduction_mode="CustOmics"
#reduction_mode="MixOmics" # reduction type
ncomp=5 # number of default components
nlatent=20 # number of default latent components
# ----------------------------------------------------------------------------------- #

# when running use optional flags
while getopts t:name:st:mode:ncomp:nlatent: flag
do
    case "${flag}" in
        t) tcga_type=${OPTARG};;
        name) mytag=${OPTARG};;
        st) label_col_name=${OPTARG};;
        mode) reduction_mode=${OPTARG};;
        ncomp) ncomp=${OPTARG};;
        nlatent) nlatent=${OPTARG};;
    esac
done
echo "Downloading TCGA Cohort(s): $tcga_type";
echo "Output Tag: $mytag";
echo "Reduction Mode: $reduction_mode";
if [ "$reduction_mode" = "MixOmics" ] ; then
    echo "Number of components: $ncomp";
fi
if [ "$reduction_mode" = "CustOmics" ] ; then
    echo "Number of latent dimensions: $nlatent";
fi


# file structure
directory="downloads"
processed="processed"
results="results"
integrated="integrated"
embeddings="integrated/embeddings"
config_fname="src/TCGA_Config-DS.json"
#config_fname="src/TCGA_Config.json"
subtypes_fname="PANCAN_Subtype_combined.csv"
omics_process=false

# create file structure if doesnt exist
subdir="$directory/TCGA-$tcga_type"
if [ ! -d "$directory" ]; then
    mkdir "$directory"
    echo "Directory '$directory' created."
fi
if [ ! -d "$subdir" ]; then
    mkdir "$subdir"
    echo "Directory '$subdir' created."
fi
if [ ! -d "$integrated" ]; then
    mkdir "$integrated"
    echo "Directory '$integrated' created."
fi
if [ ! -d "$embeddings" ]; then
    mkdir "$embeddings"
    echo "Directory '$embeddings' created."
fi
if [ ! -d "$results" ]; then
    mkdir "$results"
    echo "Directory '$results' created."
fi

# downloads data, checks if exists before download
#Rscript src/download_TCGA.r "$tcga_type" "$directory"
mydir="$directory"
if [ "$omics_process" = true ] ; then
    mydir="processed"
    if [ ! -d "$mydir" ]; then
        mkdir "$mydir"
    fi
    echo "Directory '$mydir' created."
    Rscript src/process_TCGA.r "$directory/TCGA-$tcga_type" "$mydir"
fi

# preprocessing according to several arguments:
#   directory: input directory of downloaded data
#   mytag: unique tag to add to filenames
#   processed: data to put processed data
#   path to csv file containing subtype names
#   label column in the above file to use for dim reduction
#   path to text file containing list of categorical clinical variables
#   path to text file containing list of numerical clinical variables
#   min_max: True-> Perform min_max scaling
#   var_threshs: variance filter thresholds
python src/integrate_data.py \
 "$mydir/TCGA-$tcga_type"\
 "$mytag"\
 "$integrated"\
 "$directory/TCGA-$tcga_type/$subtypes_fname"\
 "$label_col_name"\
 "$config_fname"

# perform mixomics dim reduction
# ncomp: number of components to use
if [ "$reduction_mode" = "MixOmics" ]; then
    echo "Running MixOmics Reduction"
    Rscript src/runMixOmics.r\
     "$mytag"\
     "$integrated"\
     "$embeddings"\
     "$results"\
     "$ncomp"

# performs Customics dim reduction
# nlatent: width of inner-most layer of autoencoder
# config: get numerical encoding for subtype labels
elif [ "$reduction_mode" = "CustOmics" ]; then
    echo "Running Customics Reduction"
    python src/runCustomicsMixed.py\
     "$mytag"\
     "$integrated"\
     "$embeddings"\
     "$results"\
     "$nlatent"
else
    echo "Reduction mode not recognized"
fi
