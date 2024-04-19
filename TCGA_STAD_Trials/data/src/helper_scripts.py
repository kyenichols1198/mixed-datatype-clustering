"""\
Author: Kye D Nichols
This script contains helper functions used for analysis, prep, etc.
"""
import os
import torch

import pandas as pd
import numpy as np

from customics import CustOMICS, get_common_samples, get_sub_omics_df
from sklearn.model_selection import train_test_split
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import rand_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn import preprocessing
from sklearn.manifold import TSNE
import umap

import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.offline import iplot, plot

from sparsemedoid import clustering


hidden_dim = [512, 256] # 512, 256
central_dim = [512, 256] # 512, 256
classifier_dim = [128, 64] #128, 64
lambda_classif = 5
n_epochs = 25
batch_size = 32
dropout = 0.5
beta = 1

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


# process miRNA data (RPM)
# df_cpy: input dataframe
def process_mirna(df_cpy):
    # check if rows containing mapping summary
    ncols = df_cpy.loc[df_cpy.index[0]].to_list()
    df_cpy.drop(df_cpy.index[0])
    df_cpy.columns = ncols
    to_keep = [i for i in df_cpy.index.to_list() if "reads_per_million_miRNA_mapped_" in i]
    df_cpy = df_cpy.loc[to_keep]
    df_cpy.index = [i.replace("reads_per_million_miRNA_mapped_","") for i in df_cpy.index.to_list()]
    #if log:
    #    df_cpy += 1
    #    df_cpy = np.log2(df_cpy.astype(int))
    return df_cpy

# Process reverse phase protein array data
# df_cpy: input dataframe
# summary column present
def process_protein(df_cpy, summary_col=True):
    if summary_col:
        ncols = df_cpy.loc[df_cpy.index[0]].to_list()
        df_cpy = df_cpy.drop(df_cpy.index[0:5])
        df_cpy.columns = ncols
    return df_cpy


# process methylation array data
# methyl: input dataframe
# var_thresh: variance threshold
def process_methyl(methyl,var_thresh=0.10):
    methyl = methyl.fillna(0)
    methyl = methyl.drop(methyl.index[0:2])
    methyl = rem_low_var(methyl.round(4), var_thresh = var_thresh)
    #methyl = methyl.dropna(axis="columns")
    #methyl = rem_low_var(methyl.round(4), var_thresh = var_thresh)
    return methyl

# rename components according to omics datatype
def rename_cols(indf, mystr):
    new_cols = []
    for col in indf.columns.to_list(): new_cols.append("%s-%s"%(col, mystr))
    indf.columns = new_cols
    return indf

# merge each component from Mixomics output
def merge_components(output_path, lbldf_path, input_fnames_list, new_idx):
    input_fnames = {}
    for fname in input_fnames_list:
        if fname.endswith("RNAseq.csv"):
            input_fnames["mrna"] = fname
        if fname.endswith("miRNAseq.csv"):
            input_fnames["mirna"] = fname
        if fname.endswith("Methyl.csv"):
            input_fnames["methyl"] = fname
        if fname.endswith("Protein.csv"):
            input_fnames["protein"] = fname
        if fname.endswith("CNV.csv"):
            input_fnames["cnv"] = fname
    omics_dfs = []
    for i in list(input_fnames):
        omics_df =rename_cols(pd.read_csv(input_fnames[i], index_col=0), i)
        omics_dfs.append(omics_df)
    lbldf = pd.read_csv(lbldf_path, index_col=0)
    all_df = pd.concat(omics_dfs, axis=1)
    all_df.index = new_idx
    lbldf.index = new_idx
    return all_df, lbldf

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

def save_all_plots(img_dir, cluster_num, norm_params, tag, results_path, distypes=["gower", "wishart", "podani"], kmeans=True, rand=True):
    cluster_tag = "K=%s"%str(cluster_num)
    for fname in os.listdir(results_path):
        if fname.endswith(".csv") and tag in fname and cluster_tag in fname:
            if kmedoid_score_str in fname:
                kmedoids_score = pd.read_csv(os.path.join(results_path, fname), index_col=0)
            if kmedoid_weight_str in fname:
                kmedoids_weight = pd.read_csv(os.path.join(results_path, fname), index_col=0)
            if kmedoid_lbl_str in fname:
                kmedoid_lbl = pd.read_csv(os.path.join(results_path, fname), index_col=0)
            if kmedoids_ari_str in fname:
                kmedoids_ari = pd.read_csv(os.path.join(results_path, fname), index_col=0)
            if kmedoids_ri_str in fname:
                kmedoids_ri = pd.read_csv(os.path.join(results_path, fname), index_col=0)
            if kmeans_data_str in fname:
                kmeans_results = pd.read_csv(os.path.join(results_path, fname), index_col=0)
                kmeans_ss = kmeans_results.loc[0]["Silouette"]
                kmeans_ari = kmeans_results.loc[0]["Adj Rand"]
                kmeans_ri = kmeans_results.loc[0]["Rand"]
    if kmeans:
        plot_score("Kmedoids Silouette Scores",
                    "Silouette Scores",
                    distypes,
                    kmedoids_score, norm_params,
                    kmeans=float(kmeans_ss)).write_image(os.path.join(img_dir, "K=%s_%s_Silouette.png" % (cluster_num, tag)))

        if rand:
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
                    
        if rand:

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


def runtsne(indf, imgdir, tag, labels, labels_key="labels", encoding=None):
    labels_temp = labels.copy()
    if encoding:
        encoding_rev = {encoding[k]:k for k in list(encoding)}
        labels_temp[labels_key] = labels_temp[labels_key].map(lambda s: encoding_rev.get(s))
    X = np.array(indf)
    X_embedded = TSNE(n_components = 2,
                      learning_rate = 'auto',
                      init='random',
                      perplexity = 30).fit_transform(X)
    tsne_df = pd.DataFrame({'tsne_1': X_embedded[:,0],
                            'tsne_2': X_embedded[:,1],
                            'label': labels_temp[labels_key]})
    tsne_df.index = indf.index
    fig = px.scatter(tsne_df, x="tsne_1", y="tsne_2", color='label', width=500, height=400)
    fig.write_image(os.path.join(imgdir, "%s_tnse.png" % tag))

def runumap(indf, imgdir, tag, labels, labels_key="labels", encoding=None):
    labels_temp = labels.copy()
    if encoding:
        encoding_rev = {encoding[k]:k for k in list(encoding)}
        labels_temp[labels_key] = labels_temp[labels_key].map(lambda s: encoding_rev.get(s))
    X = np.array(indf)
    X_embedded = umap.UMAP(n_neighbors=10,
                      min_dist=0.3,
                      metric='correlation').fit_transform(X)
    umap_df = pd.DataFrame({'umap_1': X_embedded[:,0],
                            'umap_2': X_embedded[:,1],
                            'label': labels_temp[labels_key]})
    umap_df.index = indf.index
    fig = px.scatter(umap_df, x="umap_1", y="umap_2", color='label', width=500, height=400)
    fig.write_image(os.path.join(imgdir, "%s_umap.png" % tag))

# calls Customics like in their Repos ipynb
def get_customics_latent(output_path, output_name, omics_df, mysamples, labels, latent_dim, encoding):
    latent_path = os.path.join(output_path, "%s_latent.csv" % output_name)
    auc_plot_path = os.path.join(output_path, "Customics_Performance.png")
    metrics_path = os.path.join(output_path, "Customics_Performance.txt")
    loss_plot_path = os.path.join(output_path, "%s_customics_loss.png" % output_name)

    samples_train, samples_test = train_test_split(mysamples, test_size=0.2)
    samples_train, samples_val = train_test_split(samples_train, test_size=0.2)

    for i in list(omics_df): omics_df[i] = omics_df[i].loc[mysamples]

    x_dim = [omics_df[i].shape[1] for i in omics_df.keys()]
    device = torch.device('cpu')
    num_classes = len(list(encoding))
    rep_dim = latent_dim
    latent_dim = latent_dim
    source_params = {}
    central_params = {'hidden_dim': central_dim,
                      'latent_dim': latent_dim,
                      'norm': True,
                      'dropout': dropout,
                      'beta': beta
                     }
    classif_params = {'n_class': num_classes,
                      'lambda': lambda_classif,
                      'hidden_layers': classifier_dim,
                      'dropout': dropout
                     }
    for i, source in enumerate(omics_df): source_params[source] = {'input_dim': x_dim[i],
                                                                   'hidden_dim': hidden_dim,
                                                                   'latent_dim': rep_dim,
                                                                   'norm': True,
                                                                   'dropout': dropout}
    train_params = {'switch': 5, 'lr': 1e-3}
    omics_train = get_sub_omics_df(omics_df, samples_train)
    omics_val = get_sub_omics_df(omics_df, samples_val)
    omics_test = get_sub_omics_df(omics_df, samples_test)

    model = CustOMICS(source_params=source_params,
                      central_params=central_params,
                      classif_params=classif_params,
                      train_params=train_params,
                      device=device
                     ).to(device)
                  
    print('Number of Parameters: ', model.get_number_parameters())
    model.fit(omics_train=omics_train,
              clinical_df=labels,
              label="labels",
              omics_val=omics_val,
              batch_size=batch_size,
              n_epochs=n_epochs,
              verbose=True
             )

    latent = model.get_latent_representation(omics_df)
    latent_df = pd.DataFrame(latent)
    latent_df.index = omics_df[list(omics_df.keys())[0]].index
    latent_df = latent_df.sort_index(ascending=False)
    latent_df.to_csv(latent_path)
    print(labels.head())
    print(np.unique(labels['labels']))
    metric = model.evaluate(omics_test=omics_test,
                            clinical_df=labels,
                            label="labels",
                            batch_size=batch_size,
                            plot_roc=True,
                            filename=auc_plot_path
                           )
    model.plot_loss(loss_plot_path)
    open(metrics_path, 'w+').write(str(metric))
    return latent_df


def rename_cols(indf, mystr):
    new_cols = []
    for col in indf.columns.to_list(): new_cols.append("%s-%s"%(col, mystr))
    indf.columns = new_cols
    return indf

def get_mixomics_output(proc_dir, output_name):
    methyldf_path = os.path.join(proc_dir, output_name + "_MixOmicsModel_methyl.csv")
    mrnadf_path = os.path.join(proc_dir, output_name + "_MixOmicsModel_RNAseq.csv")
    mirnadf_path = os.path.join(proc_dir, output_name + "_MixOmicsModel_miRNAseq.csv")
    lbldf_path = os.path.join(proc_dir, output_name + "_MixOmicsModel_labels.csv")
    methyldf = rename_cols(pd.read_csv(methyldf_path, index_col=0), "methyl")
    mrnadf = rename_cols(pd.read_csv(mrnadf_path, index_col=0), "mRNA")
    mirnadf = rename_cols(pd.read_csv(mirnadf_path, index_col=0), "miRNA")
    lbldf = pd.read_csv(lbldf_path, index_col=0)
    mixomics_df = pd.concat([methyldf, mrnadf, mirnadf], axis=1)
    comp_tag = "%s_Mixomics_Components.csv"
    mixomics_df.to_csv(os.path.join(proc_dir, comp_tag % output_name))
    lbldf = lbldf.rename(columns={"df.Y":"labels"})
    return mixomics_df, lbldf


