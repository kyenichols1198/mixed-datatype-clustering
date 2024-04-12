# ---------------------------------------------------------------------------
# Kye D Nichols
#
# Code retrieved from: https://github.com/HakimBenkirane/CustOmics
# Hakim Benkiranes code from Customics paper describing a variational autoencoder
# Compresses heterogenous multi-omics data to produce latent representation of label
# This code is the same, without survival or event tasks
# compressed into one file for educational purposes
#
# The code is described in the paper below
#
# @article{benkirane2023,
#    doi = {10.1371/journal.pcbi.1010921},
#    author = {Benkirane, Hakim AND Pradat, Yoann AND Michiels, Stefan AND Cournède, Paul-Henry},
#    journal = {PLOS Computational Biology},
#    publisher = {Public Library of Science},
#    title = {CustOmics: A versatile deep-learning based strategy for multi-omics integration},
#    year = {2023},
#    month = {03},
#    volume = {19},
#    url = {https://doi.org/10.1371/journal.pcbi.1010921},
#    pages = {1-19},
#    number = {3}
#}
# ---------------------------------------------------------------------------

import numpy as np
import pandas as pd
import shap

import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import Dataset

from scipy import stats
from scipy.stats import skew

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.manifold import TSNE
from collections import OrderedDict


class CustOMICS(nn.Module):
    def __init__(self, source_params, central_params, classif_params, train_params, device):
        super(CustOMICS, self).__init__()
        self.n_source = len(list(source_params.keys()))
        self.device = device
        self.lt_encoders = [Encoder(input_dim=source_params[source]['input_dim'], hidden_dim=source_params[source]['hidden_dim'],
                             latent_dim=source_params[source]['latent_dim'], norm_layer=source_params[source]['norm'], 
                             dropout=source_params[source]['dropout']) for source in source_params.keys()]
        self.lt_decoders = [Decoder(latent_dim=source_params[source]['latent_dim'], hidden_dim=source_params[source]['hidden_dim'],
                             output_dim=source_params[source]['input_dim'], norm_layer=source_params[source]['norm'], 
                             dropout=source_params[source]['dropout']) for source in source_params.keys()]
        self.rep_dim = sum([source_params[source]['latent_dim'] for source in source_params])
        self.central_encoder = ProbabilisticEncoder(input_dim=self.rep_dim, hidden_dim=central_params['hidden_dim'], 
                                                    latent_dim=central_params['latent_dim'], norm_layer=central_params['norm'],
                                                    dropout=central_params['dropout'])
        self.central_decoder = ProbabilisticDecoder(latent_dim=central_params['latent_dim'], hidden_dim=central_params['hidden_dim'], 
                                                    output_dim=self.rep_dim, norm_layer=central_params['norm'],
                                                    dropout=central_params['dropout'])
        self.beta = central_params['beta']
        self.num_classes = classif_params['n_class']
        self.lambda_classif = classif_params['lambda']
        self.classifier =  MultiClassifier(n_class=self.num_classes, latent_dim=central_params['latent_dim'], dropout=classif_params['dropout'],
            class_dim = classif_params['hidden_layers']).to(self.device)
        self.phase = 1
        self.switch_epoch = train_params['switch']
        self.lr = train_params['lr']
        self.autoencoders = []
        self.central_layer = None
        self._set_autoencoders()
        self._set_central_layer()
        self._relocate()
        self.optimizer = self._get_optimizer(self.lr)
        self.vae_history = []
        self.label_encoder = None
        self.one_hot_encoder = None

    def _get_optimizer(self, lr):
        lt_params = []
        for autoencoder in self.autoencoders:
            lt_params += list(autoencoder.parameters())
        lt_params += list(self.central_layer.parameters())
        lt_params += list(self.classifier.parameters())
        optimizer = Adam(lt_params, lr=lr)
        return optimizer

    def _set_autoencoders(self):
        for i in range(self.n_source):
            self.autoencoders.append(AutoEncoder(self.lt_encoders[i], self.lt_decoders[i], self.device))

    def _set_central_layer(self):
        self.central_layer = VAE(self.central_encoder, self.central_decoder, self.device)

    def _relocate(self):
        for i in range(self.n_source):
            self.autoencoders[i].to(self.device)
        self.central_layer.to(self.device)

    def _switch_phase(self, epoch):
        if epoch < self.switch_epoch:
            self.phase = 1
        else:
            self.phase = 2

    def per_source_forward(self, x):
        lt_forward = []
        for i in range(self.n_source):
            lt_forward.append(self.autoencoders[i](x[i]))
        return lt_forward

    def get_per_source_representation(self, x):
        lt_rep = []
        for i in range(self.n_source):
            lt_rep.append(self.autoencoders[i](x[i])[1])
        return lt_rep

    def decode_per_source_representation(self, lt_rep):
        lt_hat = []
        for i in range(self.n_source):
            lt_hat.append(self.autoencoders[i].decode(lt_rep[i]))
        return lt_hat

    def forward(self, x):
        lt_forward = self.per_source_forward(x)
        lt_hat = [element[0] for element in lt_forward]
        lt_rep = [element[1] for element in lt_forward]
        central_concat = torch.cat(lt_rep, dim=1)
        mean, logvar = self.central_encoder(central_concat)
        return lt_hat, lt_rep ,mean

    def _compute_loss(self, x):
        if self.phase == 1:
            lt_rep = self.get_per_source_representation(x)
            loss = 0
            for source, autoencoder in zip(x, self.autoencoders):
                loss += autoencoder.loss(source, self.beta)
            return lt_rep, loss
        elif self.phase == 2:
            lt_rep = self.get_per_source_representation(x)
            loss = 0
            for source, autoencoder in zip(x, self.autoencoders):
                loss += autoencoder.loss(source, self.beta)
            central_concat = torch.cat(lt_rep, dim=1)
            loss += self.central_layer.loss(central_concat, self.beta)
            mean, logvar = self.central_encoder(central_concat)
            z = mean
            return z, loss

    def _train_loop(self, x, labels):
        for i in range(len(x)):
            x[i] = x[i].to(self.device)
        loss = 0
        self.optimizer.zero_grad()
        if self.phase == 1:
            lt_rep, loss = self._compute_loss(x)
            for z in lt_rep:
                y_pred_proba = self.classifier(z)
                classification = classification_loss('CE', y_pred_proba, labels)
                loss += self.lambda_classif * classification

        elif self.phase == 2:
            z, loss = self._compute_loss(x)
            y_pred_proba = self.classifier(z)
            classification = classification_loss('CE', y_pred_proba, labels)
            loss += self.lambda_classif * classification

        return loss

    def fit(self, omics_train, clinical_df, label, omics_val=None, batch_size=32, n_epochs=30, verbose=False):
        
        encoded_clinical_df = clinical_df.copy()
        self.label_encoder = LabelEncoder().fit(encoded_clinical_df.loc[:, label].values)
        encoded_clinical_df.loc[:, label] = self.label_encoder.transform(encoded_clinical_df.loc[:, label].values)
        self.one_hot_encoder = OneHotEncoder(sparse_output=False).fit(encoded_clinical_df.loc[:, label].values.reshape(-1,1))

        kwargs = {'num_workers': 2, 'pin_memory': True} if self.device.type == "cuda" else {}

        lt_samples_train = get_common_samples([df for df in omics_train.values()] + [clinical_df])
        dataset_train = MultiOmicsDataset(omics_df=omics_train, clinical_df=encoded_clinical_df, lt_samples=lt_samples_train, label=label)
        train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=False, **kwargs)
        if omics_val:
            lt_samples_val = get_common_samples([df for df in omics_val.values()] + [clinical_df])
            dataset_val = MultiOmicsDataset(omics_df=omics_val, clinical_df=encoded_clinical_df, lt_samples=lt_samples_val, label=label)
            val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, **kwargs)

        self.history = []
        for epoch in range(n_epochs):
            overall_loss = 0
            self._switch_phase(epoch)
            for batch_idx, (x,labels) in enumerate(train_loader):
                self.train_all()
                loss_train = self._train_loop(x, labels)
                overall_loss += loss_train.item()
                loss_train.backward()
                self.optimizer.step()
            average_loss_train = overall_loss / ((batch_idx+1)*batch_size)
            overall_loss = 0
            if omics_val != None:
                for batch_idx, (x,labels) in enumerate(val_loader):
                    self.eval_all()
                    loss_val = self._train_loop(x, labels)
                    overall_loss += loss_val.item()
                average_loss_val = overall_loss / ((batch_idx+1)*batch_size)

                self.history.append((average_loss_train, average_loss_val))
                if verbose:
                    print("\tEpoch", epoch + 1, "complete.", "\tAverage Loss Train : ", average_loss_train, "\tAverage Loss Val : ", average_loss_val)
            else:
                self.history.append(average_loss_train)
                if verbose:
                    print("\tEpoch", epoch + 1, "complete.", "\tAverage Loss Train : ", average_loss_train)

    def get_latent_representation(self, omics_df, tensor=False):
        self.eval_all()
        if tensor == False:
            x = [torch.Tensor(omics_df[source].values) for source in omics_df.keys()]
        else:
            x = [omics for omics in omics_df]
        with torch.no_grad():
            for i in range(len(x)):
                x[i] = x[i].to(self.device)
            z, loss = self._compute_loss(x)
        return z.cpu().detach().numpy()

    def reconstruct(self, x):
        x = torch.Tensor(x)
        z = self.autoencoders[0](x)[1]
        return self.autoencoders[0].decode(z).cpu().detach().numpy()

    def plot_representation(self, omics_df, clinical_df, labels, filename, title, show=True):
        labels_df = clinical_df.loc[:, labels]
        lt_samples = get_common_samples([df for df in omics_df.values()] + [clinical_df])
        z = self.get_latent_representation(omics_df=omics_df)
        save_plot_score(filename, z, labels_df[lt_samples].values, title, show=True)

    def source_predict(self, expr_df, source):
        tensor_expr = torch.Tensor(expr_df.values)
        #tensor_expr = expr_df
        print(self.lt_encoders)
        if source == 'CNV' or source == 'protein':
            z = self.lt_encoders[0](tensor_expr)
        elif source == 'RNAseq' or source == 'miRNAseq':
            z = self.lt_encoders[1](tensor_expr)
        elif source == 'methyl':
            z = self.lt_encoders[2](tensor_expr)
        y_pred_proba = self.classifier(z)
        return y_pred_proba

    def evaluate_latent(self, omics_test, clinical_df, label, batch_size=32, plot_roc=False):
        encoded_clinical_df = clinical_df.copy()
        encoded_clinical_df.loc[:, label] = self.label_encoder.transform(encoded_clinical_df.loc[:, label].values)

        kwargs = {'num_workers': 2, 'pin_memory': True} if self.device.type == "cuda" else {}

        lt_samples_train = get_common_samples([df for df in omics_test.values()] + [clinical_df])
        dataset_test = MultiOmicsDataset(omics_df=omics_test, clinical_df=encoded_clinical_df, lt_samples=lt_samples_train, label=label)
        test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, **kwargs)

        self.eval_all()
        classif_metrics = []
        c_index = []
        with torch.no_grad():
            for batch_idx, (x, labels) in enumerate(test_loader):
                z, loss  = self._compute_loss(x)
                svc_model = SVC()
                z = self.get_latent_representation(x, tensor=True)
                svc_model.fit(z.cpu().detach().numpy(), labels.cpu().detach().numpy())
                y_pred_proba = svc_model.predict_proba()
                y_pred = torch.argmax(y_pred_proba, dim=1).cpu().detach().numpy()
                y_pred_proba = y_pred_proba.cpu().detach().numpy()
                y_true = labels.cpu().detach().numpy()
                classif_metrics.append(multi_classification_evaluation(y_true, y_pred, y_pred_proba, ohe=self.one_hot_encoder))
                if plot_roc:
                    plot_roc_multiclass(y_test=y_true, y_pred_proba=y_pred_proba, filename='test', n_classes=self.num_classes,
                                            var_names=np.unique(clinical_df.loc[:, label].values.tolist()))
                return classif_metrics


    def evaluate(self, omics_test, clinical_df, label, batch_size=32, plot_roc=False, filename=''):

        encoded_clinical_df = clinical_df.copy()
        encoded_clinical_df.loc[:, label] = self.label_encoder.transform(encoded_clinical_df.loc[:, label].values)
        kwargs = {'num_workers': 2, 'pin_memory': True} if self.device.type == "cuda" else {}
        lt_samples_train = get_common_samples([df for df in omics_test.values()] + [clinical_df])
        dataset_test = MultiOmicsDataset(omics_df=omics_test, clinical_df=encoded_clinical_df, lt_samples=lt_samples_train, label=label)
        test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, **kwargs)

        self.eval_all()
        classif_metrics = []
        c_index = []
        with torch.no_grad():
            for batch_idx, (x, labels) in enumerate(test_loader):
                z, loss  = self._compute_loss(x)
                y_pred_proba = self.classifier(z)
                y_pred = torch.argmax(y_pred_proba, dim=1).cpu().detach().numpy()
                y_pred_proba = y_pred_proba.cpu().detach().numpy()
                y_true = labels.cpu().detach().numpy()
                classif_metrics.append(multi_classification_evaluation(y_true, y_pred, y_pred_proba, ohe=self.one_hot_encoder))
                if plot_roc:
                    plot_roc_multiclass(y_test=y_true, y_pred_proba=y_pred_proba, filename=filename, n_classes=self.num_classes,
                                            var_names=np.unique(clinical_df.loc[:, label].values.tolist()))

                return classif_metrics

    def explain(self, sample_id, omics_df, clinical_df, source, subtype,label='PAM50', device='cpu', show=False):
    
        expr_df = omics_df[source]
        sample_id = list(set(sample_id).intersection(set(expr_df.index)))
        phenotype = processPhenotypeDataForSamples(clinical_df, sample_id, self.label_encoder)
        conditionaltumour=phenotype.loc[:, label] == subtype
        expr_df = expr_df.loc[sample_id,:]
        normal_expr = randomTrainingSample(expr_df, 10)
        tumour_expr = splitExprandSample(condition=conditionaltumour, sampleSize=10, expr=expr_df)
        background = addToTensor(expr_selection=normal_expr, device=device)
        male_expr_tensor = addToTensor(expr_selection=tumour_expr, device=device)
        e = shap.DeepExplainer(ModelWrapper(self, source=source), background)
        shap_values_female = e.shap_values(male_expr_tensor, ranked_outputs=None)
        shap.summary_plot(shap_values_female[0],features=tumour_expr,feature_names=list(tumour_expr.columns), show=False, plot_type="violin", max_display=10, plot_size=[4,6])
        plt.savefig('shap_{}_{}.png'.format(source, subtype), bbox_inches='tight')
        if show: plt.show()
        plt.clf()

    def plot_loss(self, png_path=None):
        n_epochs = len(self.history)
        plt.title('Evolution of the loss function with respect to the epochs')
        #plt.vlines(x=self.switch_epoch, ymin=0, ymax=2.5, colors='red', ls='--', lw=2, label='phase 2 switch')
        plt.plot(range(0, n_epochs), [loss[0] for loss in self.history], label = 'train loss')
        plt.plot(range(0, n_epochs), [loss[1] for loss in self.history], label = 'val loss')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend()
        if png_path: 
            plt.savefig(png_path)
        #plt.show()
       

    def print_parameters(self):
        lt_params = []
        lt_names = []
        for autoencoder in self.autoencoders:
            for name, param in autoencoder.named_parameters():
                lt_params.append(param.data)
                lt_names.append(name)
        for name, param in self.central_layer.named_parameters():
            lt_params.append(param.data)
            lt_names.append(name)
        print(len(lt_params))

    def get_number_parameters(self):
        sum_params = 0
        for autoencoder in self.autoencoders:
            sum_params += sum(p.numel() for p in autoencoder.parameters() if p.requires_grad)
        sum_params += sum(p.numel() for p in self.central_layer.parameters() if p.requires_grad)
        return sum_params

    def train_all(self):
        for encoder, decoder in zip(self.lt_encoders, self.lt_decoders):
            encoder.train()
            decoder.train()
        for autoencoder in self.autoencoders:
            autoencoder.train()
        self.central_layer.train()
        if self.classifier: self.classifier.train()
            
    def eval_all(self):
        for encoder, decoder in zip(self.lt_encoders, self.lt_decoders):
            encoder.eval()
            decoder.eval()
        for autoencoder in self.autoencoders:
            autoencoder.eval()
        self.central_layer.eval()
        self.classifier.eval()

def roc_auc_score_multiclass(y_true, y_pred, ohe, average = "macro"):
    y_true = ohe.transform(np.array(y_true).reshape(-1,1))
    roc_auc = roc_auc_score(y_true, y_pred, average = average, multi_class='ovo')
    return roc_auc


def multi_classification_evaluation(y_true, y_pred, y_pred_proba, average='weighted', save_confusion=False, filename=None, plot_roc=False, ohe=None):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=average)
    recall = recall_score(y_true, y_pred, average = average)
    m_f1_score = f1_score(y_true, y_pred, average = average)
    auc = roc_auc_score_multiclass(y_true, y_pred_proba, ohe, average = average)
    dt_scores = {'Accuracy': accuracy,
                'F1-score' : m_f1_score,
                'Precision' : precision,
                'Recall' : recall,
                'AUC' : auc}

    if save_confusion:
        plt.figure(figsize = (18,8))
        sns.heatmap(confusion_matrix(y_true, y_pred), annot = True, xticklabels = np.unique(y_true), yticklabels = np.unique(y_true), cmap = 'summer')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.savefig(filename + '.png')
        plt.clf()
    return dt_scores

def plot_roc_multiclass(y_test, y_pred_proba, filename="", n_classes=2, var_names=['CMML', 'MDS'], dmat=False):
    y_score = y_pred_proba
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        y_binarized = (y_test == i)
        y_scores_i = y_score[:,i]
        fpr[i], tpr[i], _ = roc_curve(y_binarized, y_scores_i)
        roc_auc[i] = auc(fpr[i], tpr[i])
    # Plot all ROC curves
    plt.figure()
    random_color = lambda : [np.random.rand() for _ in range(3)]
    #colors = [random_color() for _ in range(n_classes)]
    colors = ["red", "green", "blue", "magenta"]
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=1,
                 label='class {0} (AUC = {1:0.2f})'
                 ''.format(var_names[i], roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label="random")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve for multi-class model {}'.format(filename))
    plt.legend(loc="lower right")
    if (filename != ""):
        #plt.savefig("roc_multi_{}.png".format(filename))
        plt.savefig(filename)
    #else: #plt.show()

def save_plot_score(filename, z, y, title, show=False):
    tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
    tsne_embed = tsne.fit_transform(z)
    df = pd.DataFrame()
    df['targets'] = y
    df['x-axis'] = tsne_embed[:,0]
    df['y-axis'] = tsne_embed[:,1]
    #fashion_scatter(tsne_embed, y)
    sns_plot = sns.scatterplot(x='x-axis', y='y-axis', hue=df.targets.tolist(),
                    palette=sns.color_palette('hls', len(np.unique(y))),data=df).set(title=title)
    plt.legend(bbox_to_anchor=(1.5, 1.1), loc=2, borderaxespad=0.)
    plt.savefig(filename +  '.png', bbox_inches='tight')
    if show:
        plt.show()
    plt.clf()
    
def get_common_samples(dfs):
    lt_indices = []
    for df in dfs:
        lt_indices.append(list(df.index))
    common_indices = set(lt_indices[0])
    for i in range(1, len(lt_indices)):
        common_indices = common_indices & set(lt_indices[i])
    return list(common_indices)

def get_sub_omics_df(omics_df, lt_samples):
    return {key: value.loc[lt_samples, :] for key, value in omics_df.items()}


class MultiOmicsDataset(Dataset):
    def __init__(self, omics_df, clinical_df, lt_samples, label):
        self.omics_df = omics_df
        self.clinical_df = clinical_df
        self.lt_samples = lt_samples
        self.label = label

    def __len__(self):
        return len(self.lt_samples)

    def __getitem__(self, index):
        sample = self.lt_samples[index]
        omics_data = []
        for source, omic_df in zip(self.omics_df.keys(), self.omics_df.values()):
            omic_line = omic_df.loc[sample, :].values
            omic_line = omic_line.astype(np.float32)
            omic_line_tensor = torch.Tensor(omic_line)
            omics_data.append(omic_line_tensor)
            label = self.clinical_df.loc[sample, self.label]
        return omics_data, label

    def return_samples(self):
        return self.lt_samples


class PANCANDataset(Dataset):
    def __init__(self, omics_df, labels, cohort):
        self.omics_df = omics_df
        self.labels = labels
        self.cohort = cohort

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        omics_data = []
        for source, omic_df in zip(self.omics_df.keys(), self.omics_df.values()):
            omic_line = omic_df.iloc[index, :].values
            omic_line = omic_line.astype(np.float32)
            omic_line_tensor = torch.Tensor(omic_line)
            omics_data.append(omic_line_tensor)
        label = self.labels[index]
        return omics_data, label


def classification_loss(loss_name, y_true, y_pred ,reduction='mean'):
    if loss_name == 'BCE':
        return nn.BCEWithLogitsLoss(reduction=reduction)(y_true, y_pred)
    elif loss_name == 'CE':
        return nn.CrossEntropyLoss(reduction=reduction)(y_true, y_pred)
    else:
        raise NotImplementedError('Loss function %s is not found' % loss_name)


class AutoEncoder(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(AutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self._relocate()

    def _relocate(self):
        self.encoder.to(self.device)
        self.decoder.to(self.device)
    
    def forward(self, x):
        z = self.encoder(x)
        x_hat            = self.decoder(z)
        return x_hat, z

    def decode(self, z):
        x_hat = self.decoder(z)
        return x_hat
        
    def loss(self, x, beta):
        x_hat, z = self.forward(x)
        reconstruction_loss = nn.MSELoss()
        return reconstruction_loss(x, x_hat)


class VAE(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self._relocate()

    def _relocate(self):
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(self.device)
        z = mean + var*epsilon
        return z
                
    def forward(self, x):
        mean, log_var = self.encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))
        x_hat           = self.decoder(z)
        return x_hat, z

    def loss(self, x, beta):
        x_hat, z = self.forward(x)
        reconstruction_loss = nn.MSELoss()
        recon = reconstruction_loss(x, x_hat)
        true_samples=torch.randn(z.shape[0], z.shape[1]).to(self.device)
        MMD=torch.sum(compute_mmd(true_samples, z))

        return recon + beta * MMD


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, norm_layer=nn.BatchNorm1d, leaky_slope=0.2 ,dropout=0):
        super(Decoder, self).__init__()
        self.dt_layers = OrderedDict()
        self.dt_layers['InputLayer'] = FullyConnectedLayer(latent_dim, hidden_dim[0], norm_layer=norm_layer, leaky_slope=leaky_slope, dropout=dropout,
                                activation=True)
        block_layer_num = len(hidden_dim)
        dropout_flag = True
        for num in range(1, block_layer_num):
            self.dt_layers['Layer{}'.format(num)] = FullyConnectedLayer(hidden_dim[num - 1], hidden_dim[num], norm_layer=norm_layer, leaky_slope=leaky_slope,
                                    dropout=dropout_flag*dropout, activation=True)
            # dropout for every other layer
            dropout_flag = not dropout_flag
        # the output fully-connected layer of the classifier
        self.dt_layers['OutputLayer'] = FullyConnectedLayer(hidden_dim[-1], output_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout=0,
                                 activation=False, normalization=False)
        self.net = nn.Sequential(self.dt_layers)
    def forward(self, x):
        x_hat = self.net(x)
        return x_hat


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, norm_layer=nn.BatchNorm1d, leaky_slope=0.2, dropout=0, debug=False):
        super(Encoder, self).__init__()
        self.dt_layers = OrderedDict()
        self.dt_layers['InputLayer'] = FullyConnectedLayer(input_dim, hidden_dim[0], norm_layer=norm_layer, leaky_slope=leaky_slope, dropout=dropout,
                                activation=True)
        block_layer_num = len(hidden_dim)
        dropout_flag = True
        for num in range(1, block_layer_num):
            self.dt_layers['Layer{}'.format(num)] = FullyConnectedLayer(hidden_dim[num - 1], hidden_dim[num], norm_layer=norm_layer, leaky_slope=leaky_slope,
                                    dropout=dropout_flag*dropout, activation=True)
            dropout_flag = not dropout_flag
        self.dt_layers['OutputLayer']= FullyConnectedLayer(hidden_dim[-1], latent_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout=0,
                                 activation=False, normalization=False)
        self.net = nn.Sequential(self.dt_layers)
    def forward(self, x):
        h = self.net(x)
        return h
    def get_outputs(self, x):
        lt_output = []
        for layer in self.net:
            lt_output.append(layer(x))


class ProbabilisticEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, norm_layer=nn.BatchNorm1d, leaky_slope=0.2, dropout=0, debug=False):
        super(ProbabilisticEncoder, self).__init__()
        self.dt_layers = OrderedDict()
        self.dt_layers['InputLayer'] = FullyConnectedLayer(input_dim, hidden_dim[0], norm_layer=norm_layer, leaky_slope=leaky_slope, dropout=dropout,
                                activation=True)
        block_layer_num = len(hidden_dim)
        dropout_flag = True
        for num in range(1, block_layer_num):
            self.dt_layers['Layer{}'.format(num)] = FullyConnectedLayer(hidden_dim[num - 1], hidden_dim[num], norm_layer=norm_layer, leaky_slope=leaky_slope,
                                    dropout=dropout_flag*dropout, activation=True)
            dropout_flag = not dropout_flag
        self.net = nn.Sequential(self.dt_layers)

        self.mean_layer = FullyConnectedLayer(hidden_dim[-1], latent_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout=0,
                                 activation=False, normalization=False)
        self.log_var_layer = FullyConnectedLayer(hidden_dim[-1], latent_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout=0,
                                 activation=False, normalization=False)
    def forward(self, x):
        h = self.net(x)
        mean = self.mean_layer(h)
        log_var = self.log_var_layer(h)
        return mean, log_var


class ProbabilisticDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, norm_layer=nn.BatchNorm1d, leaky_slope=0.2, dropout=0, debug = False):
        super(ProbabilisticDecoder, self).__init__()
        self.dt_layers = OrderedDict()
        self.dt_layers['InputLayer'] = FullyConnectedLayer(latent_dim, hidden_dim[0], norm_layer=norm_layer, leaky_slope=leaky_slope, dropout=dropout,
                                activation=True)
        block_layer_num = len(hidden_dim)
        dropout_flag = True
        for num in range(1, block_layer_num):
            self.dt_layers['Layer{}'.format(num)] = FullyConnectedLayer(hidden_dim[num - 1], hidden_dim[num], norm_layer=norm_layer, leaky_slope=leaky_slope,
                                    dropout=dropout_flag*dropout, activation=True)
            # dropout for every other layer
            dropout_flag = not dropout_flag

        # the output fully-connected layer of the classifier
        self.dt_layers['OutputLayer'] = FullyConnectedLayer(hidden_dim[-1], output_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout=0,
                                 activation=False, normalization=False)

        self.net = nn.Sequential(self.dt_layers)
    def forward(self, x):
        x_hat = torch.sigmoid(self.net(x))
        return x_hat

def processPhenotypeDataForSamples(clinical_df, sample_id, le):
    phenotype = clinical_df
    phenotype = phenotype.loc[sample_id, :]
    return phenotype

def randomTrainingSample(expr,sampleSize):
    randomTrainingSampleexpr = expr.sample(n=sampleSize, axis=0)
    return randomTrainingSampleexpr

def splitExprandSample(condition, sampleSize, expr):
    expr_df_T = expr
    split_expr = expr_df_T[condition]
    split_expr = split_expr.sample(n=sampleSize, axis=0)
    return split_expr

def addToTensor(expr_selection,device):
    selection = expr_selection.values.astype(dtype='float32')
    selection = torch.Tensor(selection).to(device)
    return selection

class ModelWrapper(nn.Module):
    def __init__(self, vae_model, source):
        super(ModelWrapper, self).__init__()
        self.vae_model = vae_model
        self.source = source

    def forward(self, input):
        return self.vae_model.source_predict(input, self.source)


class MultiClassifier(nn.Module):
    def __init__(self, n_class=2, latent_dim=256, norm_layer=nn.BatchNorm1d, leaky_slope=0.2, dropout=0,
                 class_dim = [128, 64]):
        super(MultiClassifier, self).__init__()
        self.dt_layers = OrderedDict()
        self.dt_layers['InputLayer'] = FullyConnectedLayer(latent_dim, class_dim[0], norm_layer=norm_layer, leaky_slope=leaky_slope, dropout=dropout,
                                activation=True)
        block_layer_num = len(class_dim)
        dropout_flag = True
        for num in range(1, block_layer_num):
            self.dt_layers['Layer{}'.format(num)] = FullyConnectedLayer(class_dim[num - 1], class_dim[num], norm_layer=norm_layer, leaky_slope=leaky_slope,
                                    dropout=dropout_flag*dropout, activation=True)
            # dropout for every other layer
            dropout_flag = not dropout_flag

        # the output fully-connected layer of the classifier
        self.dt_layers['OutputLayer'] = FullyConnectedLayer(class_dim[-1], n_class, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout=0,
                                 activation=False, normalization=False)

        self.net = nn.Sequential(self.dt_layers)
    def forward(self, x):
        y = self.net(x)
        return y

    def predict(self, x):
        return torch.max(self.forward(x), dim=1).indices

    def compile(self, lr):
        self.optimizer = Adam(self.parameters(), lr=lr)

    def fit(self, x_train, y_train, n_epochs, verbose=False):
        self.train()
        for epoch in range(n_epochs):
            overall_loss = 0
            loss = 0
            self.optimizer.zero_grad()
            y_pred = self.forward(x_train)
            loss = classification_loss('CE', y_pred, y_train)
            overall_loss += loss.item()
        
            loss.backward()
            self.optimizer.step()
            if verbose:
                print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", overall_loss)

def classification_loss(loss_name, y_true, y_pred ,reduction='mean'):
    if loss_name == 'BCE':
        return nn.BCEWithLogitsLoss(reduction=reduction)(y_true, y_pred)
    elif loss_name == 'CE':
        return nn.CrossEntropyLoss(reduction=reduction)(y_true, y_pred)
    else:
        raise NotImplementedError('Loss function %s is not found' % loss_name)


class FullyConnectedLayer(nn.Module):
    def __init__(self, input_dim, output_dim, norm_layer=nn.BatchNorm1d, leaky_slope=0.2, dropout=0.2, activation=True, normalization=True, activation_name='LeakyReLU'):
        super(FullyConnectedLayer, self).__init__()
        self.fc_block = [nn.Linear(input_dim, output_dim)]
        if normalization:
            norm_layer = nn.BatchNorm1d
            self.fc_block.append(norm_layer(output_dim))
        if 0 < dropout <= 1:
            self.fc_block.append(nn.Dropout(p=dropout))
        if activation:
            if activation_name.lower() == 'relu':
                self.fc_block.append(nn.ReLU())
            elif activation_name.lower() == 'sigmoid':
                self.fc_block.append(nn.Sigmoid())
            elif activation_name.lower() == 'leakyrelu':
                self.fc_block.append(nn.LeakyReLU(negative_slope=leaky_slope, inplace=True))
            elif activation_name.lower() == 'tanh':
                self.fc_block.append(nn.Tanh())
            elif activation_name.lower() == 'softmax':
                self.fc_block.append(nn.Softmax(dim=1))
            elif activation_name.lower() == 'no':
                pass
            else:
                raise NotImplementedError('Activation function [%s] is not implemented' % activation_name)

        self.fc_block = nn.Sequential(*self.fc_block)
    def forward(self, x):
        y = self.fc_block(x)
        return y

def compute_kernel(x, y):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1) # (x_size, 1, dim)
    y = y.unsqueeze(0) # (1, y_size, dim)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2)/float(dim)
    return torch.exp(-kernel_input) # (x_size, y_size)

def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    mmd = x_kernel.mean() + y_kernel.mean() - 2*xy_kernel.mean()
    return mmd
