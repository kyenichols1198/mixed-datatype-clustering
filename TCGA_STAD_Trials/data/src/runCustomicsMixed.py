"""\
Author: Kye D Nichols
This script contains helper functions used for analysis, prep, etc.
# Original Code: https://github.com/HakimBenkirane/CustOmics

@article{benkirane2023,
    doi = {10.1371/journal.pcbi.1010921},
    author = {Benkirane, Hakim AND Pradat, Yoann AND Michiels, Stefan AND Cournède, Paul-Henry},
    journal = {PLOS Computational Biology},
    publisher = {Public Library of Science},
    title = {CustOmics: A versatile deep-learning based strategy for multi-omics integration},
    year = {2023},
    month = {03},
    volume = {19},
    url = {https://doi.org/10.1371/journal.pcbi.1010921},
    pages = {1-19},
    number = {3}
}
"""
from customics import CustOMICS, get_common_samples, get_sub_omics_df
from sklearn.model_selection import train_test_split
from helper_scripts import *
import os, json, sys, pickle
import torch
import pandas as pd
import numpy as np

###################### Hyperparameters ################
hidden_dim = [512, 256] # list of neurones of hidden layers of autoencoder (intermediate)
central_dim = [512, 256] # list of neurones of hidden layers of autoencoder (central)
classifier_dim = [128, 64] # list of neurones for the classifier hidden layers
lambda_classif = 5 # weight of the classification loss
n_epochs = 25 
batch_size = 32
dropout = 0.5
beta = 1
#######################################################


# calls Customics like in their Repos ipynb
def get_customics_latent(output_path, output_name, omics_df, mysamples, labels, latent_dim, num_classes):
    latent_path = os.path.join(output_path, "%s_latent.csv" % output_name)
    auc_plot_path = os.path.join(output_path, "Customics_Performance.png")
    metrics_path = os.path.join(output_path, "Customics_Performance.txt")
    loss_plot_path = os.path.join(output_path, "%s_customics_loss.png" % output_name)
    samples_train, samples_test = train_test_split(mysamples, test_size=0.2)
    samples_train, samples_val = train_test_split(samples_train, test_size=0.2)

    for i in list(omics_df): omics_df[i] = omics_df[i].loc[mysamples]

    x_dim = [omics_df[i].shape[1] for i in omics_df.keys()]
    device = torch.device('cpu')
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
    metric = model.evaluate(omics_test=omics_test,
                            clinical_df=labels,
                            label="labels",
                            batch_size=batch_size,
                            plot_roc=True,
                            filename=auc_plot_path
                           )
    model.plot_loss(loss_plot_path)
    model_pickle_path = os.path.join(output_path, '%s_Customics.pickle' % output_name)
    with open(model_pickle_path, 'wb') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    open(metrics_path, 'w+').write(str(metric))
    return latent_df



def main():
    if len(sys.argv) != 6:
        print("Wrong number of commands")
        return
    output_name = sys.argv[1] # output tag to add to fname
    proc_dir = sys.argv[2] # input directory containing processed/integrated data
    embedding_dir = sys.argv[3] # output directory to put embedding
    results_dir = sys.argv[4] # results directory to put images
    latent_dim = int(sys.argv[5]) # number of latent dimensions
    output_path =os.path.join(embedding_dir, "%s_Customics_latent.csv"%output_name)
    
    # check if the file already exists
    if not os.path.exists(output_path):
        pickle_path = os.path.join(proc_dir, "%s.pickle"%output_name)
        to_save = pickle.load(open(pickle_path, "rb"))
        # encode labels with numerical label
        encoding = {el:idx for idx, el in enumerate(np.unique(to_save["labels"]))}
        labels = to_save["labels"].map(lambda s: encoding.get(s))
        print(labels)
        mysamples = labels.index.to_list()
        omics_df = {dt:to_save[dt] for dt in list(to_save) if dt != "labels"}
        encoding_rev = {encoding[idx]: idx for idx in list(encoding)}
        # get the latent dimensions as dataframe
        latent_df = get_customics_latent(results_dir,
                                         output_name,
                                         omics_df,
                                         mysamples,
                                         labels,
                                         latent_dim,
                                         len(list(encoding_rev))
                                         )

        latent_df.columns = ["latent-%s"%str(i) for i in latent_df.columns.to_list()]
        latent_df.to_csv(output_path)


if __name__ == "__main__":
    main()
