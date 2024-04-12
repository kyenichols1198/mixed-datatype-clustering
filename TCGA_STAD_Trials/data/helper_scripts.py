"""\
Author: Kye D Nichols
This script preps TCGA data for downstream analysis

Usage: prep_data.py
"""
import os
import math
import pickle
import random
import umap
import torch

import pandas as pd
import numpy as np

from customics import CustOMICS, get_common_samples, get_sub_omics_df
from run_kmedoids import run_kmedoids_clustering

from sklearn.model_selection import train_test_split
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import rand_score
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.manifold import TSNE

import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

from run_kmedoids import run_kmedoids_clustering
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import rand_score
from customics import CustOMICS, get_common_samples, get_sub_omics_df
from helper_scripts import *
import umap
from tableone import TableOne, load_dataset
from plotly.offline import iplot, plot

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
def do_pca(df, token, n_comp):
    transf = PCA(n_components=n_comp, svd_solver='arpack').fit_transform(df)
    transf = pd.DataFrame(transf, columns=["%s-%i" % (token, i) for i in range(0,n_comp)])
    transf.index = df.index
    return transf

# PCA of weighted blocks of multiomics data
def pca_multi_omics(omics_df, pca_dims):
    pca_data = do_pca(omics_df["RNAseq"], "RNAseq", pca_dims)*3
    pca_data = pca_data.join(do_pca(omics_df["methyl"], "methyl", pca_dims)*3)
    pca_data = pca_data.join(do_pca(omics_df["miRNAseq"], "miRNAseq", pca_dims))
    if "CNV" in list(omics_df.keys()):
        pca_data = pca_data.join(do_pca(omics_df["CNV"], "CNV", pca_dims))
    if "Protein" in list(omics_df.keys()):
        pca_data = pca_data.join(do_pca(omics_df["Protein"], "Protein", pca_dims))
    return pca_data.sort_index(ascending=False)

# process miRNA data (RPM)
def process_mirna(df_cpy):
    ncols = df_cpy.loc[df_cpy.index[0]].to_list()
    df_cpy.drop(df_cpy.index[0])
    df_cpy.columns = ncols
    to_keep = [i for i in df_cpy.index.to_list() if "reads_per_million_miRNA_mapped_" in i]
    df_cpy = df_cpy.loc[to_keep]
    df_cpy.index = [i.replace("reads_per_million_miRNA_mapped_","") for i in df_cpy.index.to_list()]
    df_cpy += 1
    return np.log2(df_cpy.astype(int))

# Process reverse phase protein array data
def process_protein(df_cpy):
    ncols = df_cpy.loc[df_cpy.index[0]].to_list()
    df_cpy = df_cpy.drop(df_cpy.index[0:5])
    df_cpy.columns = ncols
    return df_cpy

# process methylation array data
def process_methyl(methyl, var_thresh=0.08):
    methyl = methyl.fillna(0)
    methyl = methyl.drop(methyl.index[0:2])
    methyl = rem_low_var(methyl.round(4), var_thresh = var_thresh)
    return methyl

# merge each component from Mixomics output
def merge_components(output_path, input_fnames_list):
    merged_df=None
    input_fnames = {}
    for fname in input_fnames_list:
        if fname.endswith("RNAseq.csv"):
            input_fnames["mrna"] = fname
        if fname.endswith("miRNAseq.csv"):
            input_fnames["mirna"] = fname
        if fname.endswith("methyl.csv"):
            input_fnames["methyl"] = fname
        if fname.endswith("Protein.csv"):
            input_fnames["protein"] = fname
        if fname.endswith("CNV.csv"):
            input_fnames["cnv"] = fname
    for idx, i in enumerate(list(input_fnames)):
        inpath = os.path.join(output_path, input_fnames[i])
        indf = pd.read_csv(inpath, index_col=0)
        indf.columns = ["%s-%s"%i for (i,j) in indf.columns.to_list()]
        if idx>0: merged_df = indf
        else:  merged_df = merged_df.join(indf)
    return merged_df

# plot silouette score
def plot_score(title, y_axis_str, id_strs, scores_df, norm_param, kmeans=None, font_size = 20, width=750, height=400):
    plot_dict= {"norm param":norm_param}; ys = []
    for id_str in id_strs:
        select_cols = []
        for col in scores_df.columns:
            if id_str in col:
                select_cols.append(col)
        y_str = "%s " % id_str
        ys.append(y_str)
        plot_dict[y_str] = scores_df[select_cols].loc[0].to_list()
    score_plot_df = pd.DataFrame().from_dict(plot_dict)
    fig = px.line(score_plot_df, x="norm param", y=ys, width=width, height=height)
    if kmeans: fig.add_hline(y=kmeans, line_dash='dash', line_color='gray',annotation_text= 'kmeans')
    fig.update_layout(yaxis = dict(tickfont = dict(size=font_size)))
    fig.update_layout(xaxis = dict(tickfont = dict(size=font_size)))
    fig.update_layout(xaxis_title="Normalization Parameter", yaxis_title=y_axis_str)
    fig.update_layout(yaxis_title = dict(font = dict(size=font_size)))
    fig.update_layout(xaxis_title = dict(font = dict(size=font_size)))
    fig.update_layout(title = title)
    fig.update_layout(legend = dict(font = dict(size=font_size-5)))
    fig.update_layout(title = dict(font = dict(size=font_size+2)))
    fig.update_traces(line={'width': 6})
    return fig

# normalize feature weights
def norm_feature_weights(weights_df):
    for col in weights_df.columns:
        weights_df[col] = weights_df[col]/max(weights_df[col])
    return weights_df
   
# plot Rand Index or adjusted
#def plot_rand_index(id_str, df, norm_param, scoring):
#    select_cols = []
#    for col in scores_df.columns:
#        if id_str in str(col):
#            select_cols.append(col)
#            #print("%s: %f" % (col, scores_df[col][0]))
#    _plot_df = pd.DataFrame().from_dict({("%s"%scoring): df[select_cols].loc[0], "norm param":normalization_param})
#    px.lineplot(data=_plot_df, x="norm param", y=("%s"%scoring)).set_title("%s_%s"%(id_str, scoring))

# plot heatmap of norm weights
def plot_heatmap(title, weights_df, id_strs, height=1200):
    select_cols = []
    for id_str in id_strs:
        for col in weights_df.columns:
            if id_str in col:
                select_cols.append(col)

    layout = go.Layout(width=1000, height=height)
    data = go.Heatmap(z=np.array(weights_df[select_cols]),
                     x=[i for i in weights_df[select_cols].columns.to_list()],
                     y=weights_df[select_cols].index.to_list(),
                     colorscale = 'Cividis')
    fig = go.Figure(data=[data], layout=layout)
    fig.update_layout(yaxis = dict(tickfont = dict(size=15)))
    fig.update_layout(title = title)
    fig.update_layout(legend = dict(font = dict(size=15)))
    fig.update_layout(title = dict(font = dict(size=17)))
    return fig

# calculate adjusted and unadjusted Rand scores
def get_rand_index(cluster_df, lbls):
    out_dict_rand={}; out_dict_arand={}
    for col in cluster_df.columns:
        out_dict_rand[col]=[rand_score(cluster_df[col], lbls)]
        out_dict_arand[col]=[adjusted_rand_score(cluster_df[col], lbls)]
    return pd.DataFrame().from_dict(out_dict_rand), pd.DataFrame().from_dict(out_dict_arand)

# plot Rand Index or adjusted
def plot_rand(id_strs, scores_df, norm_param):
    plot_dict= {"norm param":norm_param}; ys = []
    for id_str in id_strs:
        select_cols = []
        for col in scores_df.columns:
            if id_str in col:
                select_cols.append(col)
        y_str = id_str
        ys.append(y_str)
        plot_dict[y_str] = scores_df[select_cols].loc[0].to_list()
    score_plot_df = pd.DataFrame().from_dict(plot_dict)
    fig = px.line(score_plot_df, x="norm param", y=ys, width=750, height=400)
    return fig

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

kmedoid_score_str = "kmedoids_scores"
kmedoid_weight_str = "kmedoids_feature_weights"
kmedoid_lbl_str = "kmedoids_cluster_labels"
kmeans_data_str = "kmeans_data"
kmedoids_ari_str = "kmedoids_adjusted_rand"
kmedoids_ri_str = "kmedoids_rand"

def get_clustering_results(input_path, output_dir, tag, cluster_num, distance_types, norm_params, input_df, lbls, kmeans_run=True):
    scores_df, cluster_labels_df, feature_weights_df = run_kmedoids_clustering([cluster_num],
                                                                                distance_types,
                                                                                norm_params,
                                                                                input_df)
    scores_df.to_csv(os.path.join(output_dir, "%s_%s_K=%i.csv" % (tag, kmedoid_score_str, cluster_num)))
    feature_weights_df.to_csv(os.path.join(output_dir, "%s_%s_K=%i.csv" % (tag, kmedoid_weight_str, cluster_num)))
    cluster_labels_df.to_csv(os.path.join(output_dir, "%s_%s_K=%i.csv" % (tag, kmedoid_lbl_str, cluster_num)))
    if kmeans_run:
        kmeans = KMeans(n_clusters=cluster_num).fit(np.array(input_df))
        kmeans_lbls = kmeans.labels_
        kmeans_scores = silhouette_score(np.array(input_df), kmeans.fit_predict(np.array(input_df)))
        kmeans_adj_rand = adjusted_rand_score(kmeans_lbls, lbls["labels"].to_list())
        kmeans_rand = rand_score(kmeans_lbls, lbls["labels"].to_list())
        pd.DataFrame().from_dict({"Silouette":[kmeans_scores],
                                  "Adj Rand":[kmeans_adj_rand],
                                  "Rand":[kmeans_rand]}).to_csv(os.path.join(output_dir,"%s_%s_K=%i.csv"% (tag, kmeans_data_str, cluster_num)))
    adj_rand_dict = {}; rand_dict = {}
    for col in cluster_labels_df.columns:
        adjr = adjusted_rand_score(cluster_labels_df[col].to_list(), lbls["labels"].to_list())
        adj_rand_dict[col] = [adjr]
        rs = rand_score(cluster_labels_df[col].to_list(), lbls["labels"].to_list())
        rand_dict[col] = [rs]
    pd.DataFrame().from_dict(adj_rand_dict).to_csv(os.path.join(output_dir,"%s_%s_K=%i.csv" % (tag, kmedoids_ari_str, cluster_num)))
    pd.DataFrame().from_dict(rand_dict).to_csv(os.path.join(output_dir,"%s_%s_K=%i.csv" % (tag, kmedoids_ri_str, cluster_num)))


def save_all_plots(img_dir, cluster_num, norm_params, tag, results_path, distypes=["gower", "wishart", "podani"], kmeans=True):
    cluster_tag = "K=%s"%str(cluster_num)
    for fname in os.listdir(results_path):
        if fname.endswith(".csv") and tag in fname and cluster_tag in fname:
            if kmedoid_score_str in fname:
                kmedoids_score = pd.read_csv(os.path.join(results_path, fname), index_col=0).sort_index(ascending=False)
            if kmedoid_weight_str in fname:
                kmedoids_weight = pd.read_csv(os.path.join(results_path, fname), index_col=0).sort_index(ascending=False)
            if kmedoid_lbl_str in fname:
                kmedoid_lbl = pd.read_csv(os.path.join(results_path, fname), index_col=0).sort_index(ascending=False)
            if kmedoids_ari_str in fname:
                kmedoids_ari = pd.read_csv(os.path.join(results_path, fname), index_col=0).sort_index(ascending=False)
            if kmedoids_ri_str in fname:
                kmedoids_ri = pd.read_csv(os.path.join(results_path, fname), index_col=0).sort_index(ascending=False)
            if kmeans_data_str in fname:
                kmeans_results = pd.read_csv(os.path.join(results_path, fname), index_col=0).sort_index(ascending=False)
                kmeans_ss = kmeans_results.loc[0]["Silouette"]
                kmeans_ari = kmeans_results.loc[0]["Adj Rand"]
                kmeans_ri = kmeans_results.loc[0]["Rand"]
    if kmeans:
        plot_score("Kmedoids Silouette Scores",
                    "Silouette Scores",
                    distypes,
                    kmedoids_score, norm_params,
                    kmeans=float(kmeans_ss)).write_image(os.path.join(img_dir, "K=%s_%s_Silouette.png" % (cluster_num, tag)))

        plot_score("Kmedoids Adjusted Rand Index With Subtype Labels",
                    "ARI",
                    distypes,
                    kmedoids_ari,
                    norm_params,
                    kmeans=kmeans_ari).write_image(os.path.join(img_dir, "K=%s_%s_ARI.png" % (cluster_num, tag)))
        plot_score("Kmedoids Rand Index With Subtype Labels",
                    "RI",
                    distypes,
                    kmedoids_ri,
                    norm_params,
                    kmeans=kmeans_ri).write_image(os.path.join(img_dir,"K=%s_%s_RI.png" % (cluster_num, tag)))
    else:
        plot_score("Kmedoids Silouette Scores",
                    "Silouette Scores",
                    distypes,
                    kmedoids_score, norm_params).write_image(os.path.join(img_dir,"K=%s_%s_Silouette.png" % (cluster_num, tag)))

        plot_score("Kmedoids Adjusted Rand Index With Subtype Labels",
                    "ARI",
                    distypes,
                    kmedoids_ari,
                    norm_params).write_image(os.path.join(img_dir,"K=%s_%s_ARI.png" % (cluster_num, tag)))
        plot_score("Kmedoids Rand Index With Subtype Labels",
                    "RI",
                    distypes,
                    kmedoids_ri,
                    norm_params).write_image(os.path.join(img_dir,"K=%s_%s_RI.png" % (cluster_num, tag)))
    
    plot_score("Kmedoids % Non-Zero Features", "% Non-Zero Features",
                distypes,
                get_perc_nonzero(kmedoids_weight),
                norm_params).write_image(os.path.join(img_dir,"K=%s_%s_Perc_nonZ.png" % (cluster_num, tag)))

        
    #init_notebook_mode(connected=True)
    fig = plot_heatmap("Kmedoids Normalized Weights %s K=%s" % (tag, str(cluster_num)),
                kmedoids_weight,
                distypes)
    plot(fig, filename=os.path.join(img_dir,"K=%s_%s_Feature_Map.html" % (cluster_num, tag)))


def runtsne(indf, imgdir, tag, encoding, labels, perplexity = 30, labels_key="labels"):
    encoding_rev = {encoding[k]:k for k in list(encoding)}
    X = np.array(indf)
    X_embedded = TSNE(n_components = 2,
                      learning_rate = 'auto',
                      init='random',
                      perplexity = perplexity).fit_transform(X)
    tsne_df = pd.DataFrame({'tsne_1': X_embedded[:,0],
                            'tsne_2': X_embedded[:,1],
                            'label': labels[labels_key].map(lambda s: encoding_rev.get(s))})
    tsne_df.index = indf.index
    fig = px.scatter(tsne_df, x="tsne_1", y="tsne_2", color='label')
    fig.write_image(os.path.join(imgdir, "%s_tnse.png" % tag))


def runumap(indf, imgdir, tag, encoding, labels, n_neighbors = 10, labels_key="labels"):
    encoding_rev = {encoding[k]:k for k in list(encoding)}
    X = np.array(indf)
    X_embedded = umap.UMAP(n_neighbors=n_neighbors,
                      min_dist=0.3,
                      metric='correlation').fit_transform(X)
    umap_df = pd.DataFrame({'umap_1': X_embedded[:,0],
                            'umap_2': X_embedded[:,1],
                            'label': labels[labels_key].map(lambda s: encoding_rev.get(s))})
    umap_df.index = indf.index
    fig = px.scatter(umap_df, x="umap_1", y="umap_2", color='label')
    fig.write_image(os.path.join(imgdir, "%s_umap.png" % tag))


def compare_clusters(indf, outpath, cluster_num, dist_type, norm_param, cat_cols):
    param_str = "K=%s_dist=%s_S=%s" % (cluster_num, dist_type, norm_param)
    my_lbls = lbls_df[[param_str]]
    df = indf.join(my_lbls)
    groupby = [param_str] 
    mytable1 = TableOne(indf,
                        columns=df.columnsto_list(),
                        categorical=cat_cols,
                        groupby=groupby,
                        pval=True,
                        htest_name=True)
    mytable1.to_excel(outpath)
    return mytable1

