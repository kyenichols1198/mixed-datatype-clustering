"""
Author: Kye D Nichols
This script preps TCGA data for downstream analysis

Usage: prep_data.py
"""
import os
import pandas as pd
import numpy as np
from customics import get_common_samples
from helper_scripts import *


methyl_thresh = 0.10
mrna_thresh = 1.00

# prepares downloaded data for dimensional reduciton and subsequent clustering
def prep_multi_omics(input_dir, output_name, output_path, label_col_name, label_idx, datatype_tag_dict, sep_token, labels_path, encoding):
    labels = pd.read_csv(labels_path, index_col=label_idx)[[label_col_name]]
    labels = labels.rename_axis("", axis="rows").rename(columns={label_col_name:"labels"})
    data_type_list = list(datatype_tag_dict)

    all_sets = {"labels":labels[["labels"]].dropna(axis="rows")}
    merged_data = merge_from_directory(datatype_tag_dict, input_dir, sep_token)
    print("Formating RNA-seq Data")
    mrna = merged_data["RNAseq"]
    mrna = mrna.dropna(axis="columns")
    mrna = format_barcodes(mrna)
    mrna = rem_low_var(mrna.round(4), var_thresh=mrna_thresh)
    mrna = format_barcodes(mrna)
    print(mrna.shape)
    mrna_scaled = min_max_scale(mrna)
    all_sets["RNAseq"] = mrna_scaled

    print("Formating Methylation Data")
    methyl = process_methyl(merged_data["methyl"], var_thresh=methyl_thresh)
    methyl = format_barcodes(methyl)
    print(methyl.shape)
    all_sets["methyl"] = methyl

    if "CNV" in data_type_list:
        print("Formating CNV Data")
        cnv = merged_data["CNV"]
        cnv = cnv.dropna(axis="columns")
        cnv = rem_low_var(cnv)
        cnv = format_barcodes(cnv)
        cnv.to_csv(os.path.join(output_path, '%s_cnv.csv' % output_name))
        print(cnv.shape)
        cnv_scaled = min_max_scale(cnv)
        all_sets["CNV"] = cnv_scaled

    print("Formating miRNA-seq Data")
    mirna = process_mirna(merged_data["miRNAseq"])
    mirna = mirna.dropna(axis="columns")
    mirna = rem_low_var(mirna.round(4))
    mirna = format_barcodes(mirna)
    print(mirna.shape)
    mirna_scaled = min_max_scale(mirna)
    all_sets["miRNAseq"] = mirna_scaled

    if "Protein" in data_type_list:
        print("Formating Protein Data")
        protein = process_protein(merged_data["Protein"])
        protein = protein.dropna(axis="columns")
        protein = rem_low_var(protein.round(4))
        protein = format_barcodes(protein)
        protein.to_csv(os.path.join(output_path, '%s_protein.csv' % output_name))
        print(protein.shape)
        protein_scaled = min_max_scale(protein)
        all_sets["Protein"] = protein_scaled

    labels = all_sets["labels"].dropna()[["labels"]].map(lambda s: encoding.get(s)).dropna().astype(int)
    omics_df = {k:all_sets[k].dropna(axis="columns").astype(float) for k in list(all_sets) if k!="labels"}

    mysamples = get_common_samples([labels]+list(omics_df.values()))
    labels = labels.loc[mysamples].sort_index(ascending=False)
    for dtype in list(omics_df): omics_df[dtype] = omics_df[dtype].loc[mysamples].sort_index(ascending=False)

    encoding_rev = {encoding[k]:k for k in list(encoding)}
    to_save = omics_df.copy(); to_save["labels"] = labels.copy().map(lambda s: encoding_rev.get(s))
    with open(os.path.join(output_path, '%s.pickle' % output_name), 'wb') as handle:
        pickle.dump(to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    outpaths = []
    for df_key in list(to_save):
        outpath = os.path.join(output_path,'%s_%s.csv'% (output_name, df_key))
        outpaths.append(outpath)
        to_save[df_key].to_csv(outpath)
        
    return (omics_df, labels, mysamples, outpaths)


