"""\
Author: Kye D Nichols
This script preps TCGA data for downstream analysis

Usage: prep_data.py
"""
import os, torch, pickle
import pandas as pd
import numpy as np
from customics import CustOMICS, get_common_samples, get_sub_omics_df
from sklearn.model_selection import train_test_split
from helper_scripts import *

hidden_dim = [512, 256] # 512, 256
central_dim = [512, 256] # 512, 256
classifier_dim = [128, 64] #128, 64
lambda_classif = 5
n_epochs = 25
batch_size = 32
dropout = 0.5
beta = 1

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

    metric = model.evaluate(omics_test=omics_test,
                            clinical_df=labels,
                            label="labels",
                            batch_size=batch_size,
                            plot_roc=False,
                            filename=auc_plot_path
                           )
    model.plot_loss(loss_plot_path)
    open(metrics_path, 'w+').write(str(metric))
    return latent_df


