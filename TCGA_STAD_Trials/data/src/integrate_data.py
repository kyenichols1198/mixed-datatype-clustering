"""
Author: Kye D Nichols
This script preps TCGA data for downstream analysis

Usage: prep_data.py
"""
import os, sys, json
import pandas as pd
import numpy as np
from helper_scripts import *


def main():
    var_thresh_dict = {}
    encoding = {}
    datatype_tag_dict = {}
    num_cols = []
    # Check if the correct number of arguments is provided
    if len(sys.argv) != 7:
        print("Prep data: wrong number of arguments")
        return
    # Retrieve command-line arguments
    input_dir = sys.argv[1]
    output_name = sys.argv[2]
    proc_dir= sys.argv[3]
    labels_path = sys.argv[4]
    label_col_name = sys.argv[5]
    config_fname = sys.argv[6]

    with open(config_fname, 'r') as openfile:
        json_file = json.load(openfile)
        print(json_file)
        datatype_tag_dict = json_file['datatypes']
        encoding_dict = json_file['encodings']
        encoding_list = encoding_dict[label_col_name]
        encodings = {idx:el for idx, el in enumerate(encoding_list)}
        cat_cols = json_file['categorical']
        num_cols = json_file['numerical']
        (omics_df, labels, mysamples, outpaths) = prep_multi_omics(input_dir,
                                                               output_name,
                                                               proc_dir,
                                                               labels_path,
                                                               label_col_name,
                                                               datatype_tag_dict,
                                                               encodings,
                                                               )
        open(os.path.join(proc_dir, "%s-sample_list.txt"%output_name), 'w+').write("%s\n" % ("\n".join(mysamples)))

if __name__ == "__main__":
    main()
