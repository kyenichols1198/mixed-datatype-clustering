"""\
Author: Kye D Nichols
This script contains helper functions used for integration and prep, etc...
Note: some methods aren't used
"""
import os
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn import preprocessing

# label encoder
def one_hot_encode(df, cat_columns, norm=False):
    df_encoded = df.copy()
    label_encoders = {}
    for col in cat_columns:
        label_encoders[col] = LabelEncoder()
        df_encoded[col] = label_encoders[col].fit_transform(list(df[col]))
    if norm: df_encoded = min_max_scale(df_encoded)
    return df_encoded

# number of clusters (int), list of distance types ([str]), ...
# from J's run.py code in test subdirectory
def run_kmedoids_clustering(clusters, distance_types, normalization_param, X_df):
    total_runs = (len(clusters) * len(distance_types) * len(normalization_param))
    Scores = np.zeros((1, total_runs))
    barcodes = X_df.index.to_list()
    X = X_df.to_numpy()
    P = X.shape[0]
    N = X.shape[1]
    prefix_cols = []
    all_feature_weights = np.zeros((N, total_runs))
    all_cluster_labels = np.zeros((P, total_runs))
    iter1 = 0
    for K in clusters:
        for distance in distance_types:
            for S in normalization_param:
                results_path_prefix = f"K={K}_dist={distance}_S={S}"
                prefix_col = f"N={N}_K={K}_dist={distance}_nparam={S}"
                prefix_cols.append(results_path_prefix)
                (
                    cluster_labels,
                    feature_weights,
                    feature_order,
                    weighted_distances,
                ) = clustering.sparse_kmedoids(
                    X,
                    distance_type=distance,
                    k=K,
                    s=S,
                    max_attempts=6,
                    method="pam",
                    init="build",
                    max_iter=100,
                    random_state=None,
                )
                Scores[0, iter1] += silhouette_score(
                    weighted_distances, cluster_labels, metric="precomputed"
                )
                all_feature_weights[:, iter1] = feature_weights
                all_cluster_labels[:, iter1] = cluster_labels
                iter1 += 1
    feature_weights_df = pd.DataFrame(all_feature_weights)
    cluster_labels_df = pd.DataFrame(all_cluster_labels)
    cluster_labels_df.index = barcodes
    cluster_labels_df.columns = prefix_cols
    feature_weights_df.index = X_df.columns.to_list()
    feature_weights_df.columns = prefix_cols
    scores_df = pd.DataFrame(Scores)
    scores_df.columns = prefix_cols
    return scores_df, cluster_labels_df, feature_weights_df

# combines all omics data together if seperate data
def merge_from_directory(dtag_dict, input_dir, sep_token):
    merged_dict = {}
    for dtag in list(dtag_dict):
        df_list = [pd.read_csv(os.path.join(input_dir, fname), sep=sep_token, index_col=0).T for fname in os.listdir(input_dir) if fname.endswith(dtag_dict[dtag])]
        merged_dict[dtag] = pd.concat(df_list)
    return merged_dict

# low variance filter. default is zero if no variability
def rem_low_var(input_df, var_thresh = 0.0):
    names = input_df.var().keys()
    columns_to_keep = []
    for idx,col_var in enumerate(input_df.var().to_list()):
        name = names[idx]
        if col_var > var_thresh: columns_to_keep.append(name)
    return input_df[columns_to_keep]

# format barcodes to patient level
def format_barcodes(m_df):
    ex_code = m_df.index.to_list()[0]
    if "-" in ex_code: m_df.index = np.array(['-'.join(i.split("-")[:3]) for i in m_df.index])
    else: m_df.index = np.array(['-'.join(i.split(".")[:3]) for i in m_df.index])
    m_df = m_df[~m_df.index.duplicated(keep='first')]
    return m_df

# min-max scaling and reapplying barcode
def min_max_scale(m_df):
    m_mat = preprocessing.MinMaxScaler().fit_transform(m_df)
    ret_df = pd.DataFrame(m_mat, columns = m_df.columns)
    ret_df.index = m_df.index
    return ret_df

# standard scaling and reapplying barcode (as an option)
def standard_scale(m_df):
    m_mat = preprocessing.StandardScaler().fit_transform(m_df)
    ret_df = pd.DataFrame(m_mat, columns = m_df.columns)
    ret_df.index = m_df.index
    return ret_df

# PCA of multiomics data blocks
# input dataframe: df
# datatype: dt
# number of components: n_comp
def do_pca(df, dt, n_comp):
    transf = PCA(n_components=n_comp, svd_solver='arpack').fit_transform(df)
    transf = pd.DataFrame(transf, columns=["%s-%i" % (dt, i) for i in range(0,n_comp)])
    transf.index = df.index
    return transf

# PCA of weighted blocks of multiomics data
# input dataframe: omics_df
# number of components: n_comp
def pca_multi_omics(omics_df, pca_dims=200):
    pca_data = do_pca(omics_df["RNAseq"], "RNAseq", pca_dims)*3
    pca_data = pca_data.join(do_pca(omics_df["methyl"], "methyl", pca_dims)*3)
    pca_data = pca_data.join(do_pca(omics_df["miRNAseq"], "miRNAseq", pca_dims))
    if "CNV" in list(omics_df.keys()):
        pca_data = pca_data.join(do_pca(omics_df["CNV"], "CNV", pca_dims))
    if "Protein" in list(omics_df.keys()):
        pca_data = pca_data.join(do_pca(omics_df["Protein"], "Protein", pca_dims))
    return pca_data

# rename components according to omics datatype
def rename_cols(indf, mystr):
    new_cols = []
    for col in indf.columns.to_list(): new_cols.append("%s-%s"%(col, mystr))
    indf.columns = new_cols
    return indf

# normalize feature weights
def norm_feature_weights(weights_df):
    for col in weights_df.columns:
        weights_df[col] = weights_df[col]/max(weights_df[col])
    return weights_df

# get percentage of zero weights
def get_perc_nonzero(kmedoids_weight):
    kmedoids_weight_norm = norm_feature_weights(kmedoids_weight)
    perc_nonz = {}
    len_index = len(kmedoids_weight_norm.index.to_list())
    for col in kmedoids_weight_norm.columns.to_list():
        num_nonz = len([i for i in kmedoids_weight_norm[col] if i > 0.0001])
        perc = (num_nonz/len_index)*100 
        perc_nonz[col]=[perc]
    
    kmedoids_pnz = pd.DataFrame().from_dict(perc_nonz)
    kmedoids_pnz.index = [0]
    return kmedoids_pnz

# prepares downloaded data for dimensional reduciton and subsequent clustering
def prep_multi_omics(input_dir,
                     output_name,
                     output_path,
                     labels_path,
                     label_col_name,
                     datatype_dict,
                     encoding,
                     min_max=True,
                     overwrite=False):
                     
    datatype_tag_dict = {i:datatype_dict[i][0] for i in list(datatype_dict)}
    var_thresh_dict ={i:datatype_dict[i][1] for i in list(datatype_dict)}
                     
    pickle_path = os.path.join(output_path, '%s.pickle' % output_name)
    if overwrite == True or not os.path.exists(pickle_path):
        labels = pd.read_csv(labels_path, index_col=0)[[label_col_name]]
        #print("HELLO")
        #print(labels.head())
        labels = format_barcodes(labels.rename_axis("", axis="rows").rename(columns={label_col_name:"labels"}))
        #print(labels.head())
        data_type_list = list(datatype_tag_dict)
        all_sets = {"labels":labels}
        merged_data = merge_from_directory(datatype_tag_dict, input_dir, sep_token=",")
        print("Formating RNA-seq Data")
        
        mrna = merged_data["RNAseq"]
        rna_thresh = var_thresh_dict["RNAseq"]
        mrna = mrna.dropna(axis="columns")
        # select gene-level as opposed to transcript level
        mrna = format_barcodes(mrna)
        mrna = rem_low_var(mrna, var_thresh=rna_thresh)
        mrna = format_barcodes(mrna)
        print(mrna.shape)
        if min_max: mrna = min_max_scale(mrna)
        all_sets["RNAseq"] = mrna
        
        if "methyl" in data_type_list:
            print("Formating Methylation Data")
            methyl = merged_data["methyl"]
            methyl = methyl.drop(methyl.index[0:2])
            methyl_thresh = var_thresh_dict["methyl"]
            methyl = rem_low_var(methyl, var_thresh = methyl_thresh)
            to_keep = []
            for i in methyl.columns.to_list():
                if not methyl[i].isna().sum() > 0: to_keep.append(i)
            methyl = methyl[to_keep]
            methyl = methyl.fillna(methyl.mean())
            methyl = format_barcodes(methyl)
            print(methyl.shape)
            # if min_max: methyl = min_max_scale(methyl) already scaled
            all_sets["methyl"] = methyl

        if "CNV" in data_type_list:
            print("Formating CNV Data")
            cnv = merged_data["CNV"]
            cnv = cnv.dropna(axis="columns")
            cnv = rem_low_var(cnv)
            cnv = format_barcodes(cnv)
            cnv.to_csv(os.path.join(output_path, '%s_cnv.csv' % output_name))
            print(cnv.shape)
            if min_max: cnv = min_max_scale(cnv)
            all_sets["CNV"] = cnv

        if "miRNAseq" in data_type_list:
            print("Formating miRNA-seq Data")
            mirna = merged_data["miRNAseq"]
            ncols = mirna.loc[mirna.index[0]].to_list()
            mirna= mirna.drop(mirna.index[0])
            mirna.columns = ncols
            to_keep = [i for i in mirna.index.to_list() if "reads_per_million_miRNA_mapped_" in i]
            mirna = mirna.loc[to_keep]
            mirna.index = [i.replace("reads_per_million_miRNA_mapped_","") for i in mirna.index.to_list()]
            mirna = mirna.dropna(axis="columns")
            mirna = rem_low_var(mirna)
            mirna = format_barcodes(mirna)
            print(mirna.shape)
            print(mirna.head())
            if min_max: mirna = min_max_scale(mirna)
            all_sets["miRNAseq"] = mirna

        print(all_sets["labels"])
        print(encoding)
        #labels = all_sets["labels"][["labels"]].map(lambda s: encoding.get(s)).dropna().astype(int)
        #print(labels.head())
        #print("hello")
        omics_df = {k:all_sets[k].dropna(axis="columns").astype(float) for k in list(all_sets) if k!="labels"}
        encoding_rev = {encoding[k]:k for k in list(encoding)}

        mysamples = get_common_samples([labels.map(lambda s: encoding_rev.get(s)).dropna()]+list(omics_df.values()))
        labels = labels.loc[mysamples].sort_index(ascending=False)
        for dtype in list(omics_df): omics_df[dtype] = omics_df[dtype].loc[mysamples].sort_index(ascending=False)

        to_save = omics_df.copy()
        #to_save["labels"] = labels.copy().map(lambda s: encoding_rev.get(s))
        to_save["labels"] = labels.copy()
        with open(pickle_path, 'wb') as handle:
            pickle.dump(to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        to_save = pickle.load(open(pickle_path, "rb"))
        labels = to_save["labels"].map(lambda s: encoding.get(s))
        mysamples = labels.index.to_list()
        omics_df = {dt:to_save[dt] for dt in list(to_save) if dt != "labels"}
    outpaths = []
    for df_key in list(to_save):
        outpath = os.path.join(output_path,'%s_%s.csv'% (output_name, df_key))
        outpaths.append(outpath)
        to_save[df_key].to_csv(outpath)
    return (omics_df, labels, mysamples, outpaths)

def get_common_samples(dfs):
    lt_indices = []
    for df in dfs:
        lt_indices.append(list(df.index))
    common_indices = set(lt_indices[0])
    for i in range(1, len(lt_indices)):
        common_indices = common_indices & set(lt_indices[i])
    return list(common_indices)
