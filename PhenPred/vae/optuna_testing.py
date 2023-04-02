#pip install optuna
#DATA
import pandas as pd
import numpy as np


#MODELS
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate

import optuna
from optuna.trial import TrialState

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from scipy import stats


#SEEDS
import random
import warnings


#TIMESTAMP
import calendar
import time


if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")


seed = 7
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore')

transcriptomics = pd.read_csv("/data/benchmarks/clines/transcriptomics.csv", index_col=0)
proteomics = pd.read_csv("/data/benchmarks/clines/proteomics.csv", index_col=0)
metabolomics = pd.read_csv("/data/benchmarks/clines/metabolomics.csv", index_col=0)
methylation = pd.read_csv("/data/benchmarks/clines/methylation.csv", index_col=0)
crisprcas9 = pd.read_csv("/data/benchmarks/clines/crisprcas9_22Q2.csv", index_col=0)
drugresponse = pd.read_csv("/data/benchmarks/clines/drugresponse.csv", index_col=0)

#Remove Features > 95% missing values
proteomics = proteomics[proteomics.isnull().sum(axis=1)/len(proteomics.columns)*100 < 95]
drugresponse = drugresponse[drugresponse.isnull().sum(axis=1)/len(drugresponse.columns)*100 < 95]


#Remove cell lines that only have data in one dataset
cell_methylation = pd.DataFrame(methylation.columns, columns = ["cell_lines"])
cell_methylation["dataset"] = "methylation"
cell_transcriptomics = pd.DataFrame(transcriptomics.columns, columns = ["cell_lines"])
cell_transcriptomics["dataset"] = "transcriptomics"
cell_proteomics = pd.DataFrame(proteomics.columns, columns = ["cell_lines"])
cell_proteomics["dataset"] = "proteomics"
cell_metabolomics = pd.DataFrame(metabolomics.columns, columns = ["cell_lines"])
cell_metabolomics["dataset"] = "metabolomics"
cell_crisprcas9 = pd.DataFrame(crisprcas9.columns, columns = ["cell_lines"])
cell_crisprcas9["dataset"] = "crisprcas9"
cell_drugresponse = pd.DataFrame(drugresponse.columns, columns = ["cell_lines"])
cell_drugresponse["dataset"] = "drugresponse"

missing_viz = pd.merge(cell_methylation,cell_transcriptomics,on = "cell_lines", how="outer")
missing_viz = pd.merge(missing_viz,cell_proteomics,on = "cell_lines", how="outer")
missing_viz = pd.merge(missing_viz,cell_metabolomics,on = "cell_lines", how="outer")
missing_viz = pd.merge(missing_viz,cell_crisprcas9,on = "cell_lines", how="outer")
missing_viz = pd.merge(missing_viz,cell_drugresponse,on = "cell_lines", how="outer")
missing_viz.index = missing_viz.cell_lines
missing_viz = missing_viz.drop(columns=['cell_lines'])
missing_viz.columns = ["Methylation", "Transcriptomics", "Proteomics", "Metabolomics", "Crispr", "DrugResponse"]

#Remove cell lines which are only in one dataset
missing_viz = missing_viz[missing_viz.isna().sum(axis=1) < 5]
cells = list(missing_viz.index)
cells.remove("SIDM00189")
cells.remove("SIDM00650")
cells = sorted(cells)

class OmicsData(Dataset):
    def __init__(self, tran_csv_file, prot_csv_file, drug_csv_file, crispr_csv_file, meta_csv_file, methy_csv_file, cells):
        
        # Load the Dataframes
        self.df_tran = pd.read_csv(tran_csv_file, index_col=0).T
        self.df_prot = pd.read_csv(prot_csv_file, index_col=0).T
        self.df_drug = pd.read_csv(drug_csv_file, index_col=0).T
        self.df_crispr = pd.read_csv(crispr_csv_file, index_col=0).T
        self.df_meta = pd.read_csv(meta_csv_file, index_col=0).T
        self.df_methy = pd.read_csv(methy_csv_file, index_col=0).T

        #Remove Features > 95% missing values
        self.df_prot = self.df_prot.T[self.df_prot.T.isnull().sum(axis=1)/len(self.df_prot.T.columns)*100 < 95].T
        self.df_drug = self.df_drug.T[self.df_drug.T.isnull().sum(axis=1)/len(self.df_drug.T.columns)*100 < 95].T

        # Add the selected cell lines
        self.samples = cells
        self.df_tran = self.df_tran.reindex(index=self.samples)
        self.df_prot = self.df_prot.reindex(index=self.samples)
        self.df_drug = self.df_drug.reindex(index=self.samples)
        self.df_crispr = self.df_crispr.reindex(index=self.samples)
        self.df_meta = self.df_meta.reindex(index=self.samples)
        self.df_methy = self.df_methy.reindex(index=self.samples)

        #Remove features with less variability
        #self.std_transcriptomics = self.df_tran.T.std(axis=1).sort_values(ascending=False)
        #self.std_methylation = self.df_methy.T.std(axis=1).sort_values(ascending=False)
        #self.std_crispr = self.df_crispr.T.std(axis=1).sort_values(ascending=False)

        #self.df_tran = self.df_tran.T.loc[self.std_transcriptomics[self.std_transcriptomics > 0.9].index,:].T
        #self.df_methy = self.df_methy.T.loc[self.std_methylation[self.std_methylation > 0.055].index,:].T
        #self.df_crispr = self.df_crispr.T.loc[self.std_crispr[self.std_crispr > 0.097].index,:].T


        # Normalize
        self.x_tran, self.scaler_tran = self.process_df(self.df_tran)
        self.x_prot, self.scaler_prot = self.process_df(self.df_prot)
        self.x_drug, self.scaler_drug = self.process_df(self.df_drug)
        self.x_crispr, self.scaler_crispr = self.process_df(self.df_crispr)
        self.x_meta, self.scaler_meta = self.process_df(self.df_meta)
        self.x_methy, self.scaler_methy = self.process_df(self.df_methy)

        # Datasets list
        self.views = [self.x_tran, self.x_prot, self.x_drug, self.x_crispr, self.x_meta, self.x_methy]

    def process_df(self, df):
        scaler = StandardScaler()
        x = scaler.fit_transform(df)
        x = np.nan_to_num(x, copy=False)

        # Convert to tensor
        x = torch.tensor(x, dtype=torch.float)
        return x, scaler

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.x_tran[idx], self.x_prot[idx], self.x_drug[idx], self.x_crispr[idx], self.x_meta[idx], self.x_methy[idx]


tran_csv_file = "/data/benchmarks/clines/transcriptomics.csv"
prot_csv_file = "/data/benchmarks/clines/proteomics.csv"
drug_csv_file = "/data/benchmarks/clines/drugresponse.csv"
crispr_csv_file = "/data/benchmarks/clines/crisprcas9_22Q2.csv"
meta_csv_file = "/data/benchmarks/clines/metabolomics.csv"
methy_csv_file = "/data/benchmarks/clines/methylation.csv"

omics_db = OmicsData(tran_csv_file, prot_csv_file, drug_csv_file, crispr_csv_file, meta_csv_file, methy_csv_file, cells)

class BottleNeck(nn.Module):
    def __init__(self, hidden_dim, group, activation_function,probability):
        super(BottleNeck, self).__init__()

        self.activation_function = activation_function
        
        self.groups = nn.ModuleList()
        for g in range(group):
            group_layers = nn.ModuleList()
            group_layers.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // group)
                              )
            )
            group_layers.append(
                nn.Sequential(
                    nn.Dropout(p=probability)
                              )
            )
            group_layers.append(
                nn.Sequential(nn.Linear(hidden_dim // group, hidden_dim // group)
                              )
            )
            self.groups.append(group_layers)
            
            
    def forward(self, x):  
        activation = self.activation_function

        #start with the input, which in this case will be the result of the first fully connected layer
        identity = x 
        out = []

        for group_layers in self.groups:
            group_out = x

            for layer in group_layers:  
                group_out = activation(layer(group_out))
            out.append(group_out)

        #concatenate, the size should be equal to the hidden size 
        out = torch.cat(out, dim=1)

        #Why do we add here the identity
        out += identity
        
        return out
    
    
class OMIC_VAE(nn.Module):
    def __init__(self, views, hidden_dim, latent_dim, probability, group, activation_function) -> None:
        
        super(OMIC_VAE, self).__init__()

        self.activation_function = activation_function
        
        
        #-- Encoders 
        self.omics_encoders = nn.ModuleList()
        for v in views:
            self.omics_encoders.append(
                nn.Sequential(
                    nn.Linear(v.shape[1], int(hidden_dim*v.shape[1]//group*group)),
                )
            )
           
        
        #-- Bottlenecks
        self.omics_bottlenecks = nn.ModuleList()
        for v in views:
            self.omics_bottlenecks.append(BottleNeck(int(hidden_dim*v.shape[1]//group*group), group = group, activation_function = activation_function, probability= probability))
            
        
        #-- Mean
        self.mus = nn.ModuleList()
        for v in views:
            self.mus.append(
                nn.Sequential(
                    nn.Linear(int(hidden_dim*v.shape[1]//group*group), latent_dim),
                )
            )
        
        
        #-- Log-Var
        self.log_vars = nn.ModuleList()
        for v in views:
            self.log_vars.append(
                nn.Sequential(
                    nn.Linear(int(hidden_dim*v.shape[1]//group*group), latent_dim),
                )
            )
           
        
        #-- Decoders 
        self.omics_decoders = nn.ModuleList()
        for v in views:
            self.omics_decoders.append(
                nn.Sequential(
                    nn.Linear(latent_dim, int(hidden_dim*v.shape[1]//group*group)),
                    activation_function,
                    nn.Linear(int(hidden_dim*v.shape[1]//group*group), v.shape[1]),
                )
            )


    def encode(self, views):
        h_bottlenecks = []
        activation = self.activation_function
        for view, encoder, bottleneck in zip(views, self.omics_encoders, self.omics_bottlenecks):
            h_bottleneck_ = activation(encoder(view))
            h_bottleneck_ = activation(bottleneck(h_bottleneck_))
            h_bottlenecks.append(h_bottleneck_)
        return h_bottlenecks
    
    
    def mean_variance(self, h_bottlenecks):
        means, log_variances = [], []
        for h_bottleneck, mu, log_var in zip(h_bottlenecks, self.mus, self.log_vars):
            mean = mu(h_bottleneck)
            var = log_var(h_bottleneck)
            means.append(mean)
            log_variances.append(var)
        return means, log_variances
    
    
    def product_of_experts(self, means, log_variances):
        #Code taken from Integrating T-cell receptor and transcriptome for 3 large-scale 
        #single-cell immune profiling analysis
        # formula: var_joint = inv(inv(var_prior) + sum(inv(var_modalities))
        logvar_joint = torch.sum(
            torch.stack([1.0 / torch.exp(log_var) for log_var in log_variances]), dim=0
        )
        logvar_joint = torch.log(1.0 / logvar_joint)
        
        # formula: mu_joint = (mu_prior*inv(var_prior) + sum(mu_modalities*inv(var_modalities))) * var_joint, 
        # where mu_prior = 0.0
        mu_joint = torch.sum(
            torch.stack([mu / torch.exp(log_var) for mu, log_var in zip( means, log_variances)]),
            dim=0,
        )
        mu_joint = mu_joint * torch.exp(logvar_joint)

        return mu_joint, logvar_joint

    
    def calculate_sample(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(log_var)
        return mean + eps * std
    
    
    def decode(self, z):
        return [decoder(z) for decoder in self.omics_decoders]
          
    
    def forward(self, views):
        
        h_bottlenecks = self.encode(views)
        means, log_variances = self.mean_variance(h_bottlenecks)
        mu_joint, logvar_joint = self.product_of_experts(means, log_variances)
        z = self.calculate_sample(mu_joint, logvar_joint)
        views_hat = self.decode(z)
        return views_hat
    
    
def mse_kl(views_hat, views, mu_joint, logvar_joint, means, log_variances, loss_type, alpha, lambd):
    n_samples = views[0].shape[0]
    if loss_type == "mse":
      mse_loss = sum([nn.MSELoss(reduction = "sum")(x, x_hat)/x.shape[1] for x, x_hat in zip(views, views_hat)])
    elif loss_type == "smoothl1":
      mse_loss = sum([nn.SmoothL1Loss(reduction = "sum")(x, x_hat)/x.shape[1] for x, x_hat in zip(views, views_hat)])
    elif loss_type == "rmse":
      mse_loss = sum([torch.sqrt(nn.MSELoss(reduction = "sum")(x, x_hat)/x.shape[1]) for x, x_hat in zip(views, views_hat)])
    
    # Compute the KL loss
    kl_loss = 0
    for mu_joint, logvar_joint in zip(means, log_variances):
        kl_loss += -0.5 * torch.sum(1 + logvar_joint - torch.pow(mu_joint, 2) - torch.exp(logvar_joint))

    # Compute the total loss
    loss = (lambd[0] * mse_loss) / n_samples + (alpha[0] * kl_loss) / n_samples

    return loss, mse_loss / n_samples, kl_loss / n_samples


def cross_validation(data, n_folds, batch_size, model, optimizer, alpha_KL, alpha_MSE, loss_type):
    
    #Initiate Cross Validation
    cv = KFold(n_folds, shuffle=True)
    
    #Train Losses
    loss_train = []
    mse_train = []
    kl_train = []
    
    #Validation Losses
    loss_val = []
    mse_val = []
    kl_val = []
    
    #Train Losses - Dataset Specific
    mse_list = {}

    for i in range(len(data.views)):
        mse_list[f"Dataset {i}"] = []

    for train_idx, val_idx in cv.split(data):
        
        # Train Data
        data_train = torch.utils.data.Subset(data, train_idx)
        dataloader_train = DataLoader(data_train, batch_size=batch_size, shuffle=False)
        
        # Validation Data
        data_test = torch.utils.data.Subset(data, val_idx)
        dataloader_test = DataLoader(data_test, batch_size=batch_size, shuffle=False)
    
    
        
        #--- TRAINING LOOP
        model.train()
        
        # dataloader train is divided into batches
        for views in dataloader_train:
            n = views[0].size(0)

            views = [view.to(device) for view in views]

            # Forward pass to get the predictions
            views_hat = model.forward(views)
            
            # Get last layer of encoder with bottleneck
            h_bottleneck = model.encode(views)
            
            # Get means and log_vars
            means, log_variances = model.mean_variance(h_bottleneck)


            mu_joint, logvar_joint = model.product_of_experts(means, log_variances)
            
            # Calculate Losses
            loss, mse, kl = mse_kl(views_hat, views, mu_joint, logvar_joint, means, log_variances, loss_type = loss_type, alpha = alpha_KL, lambd = alpha_MSE)
            
            if loss_type == "mse":
              mse_omics = [nn.MSELoss(reduction = "sum")(x, x_hat)/x.shape[1] for x, x_hat in zip(views, views_hat)]
            elif loss_type == "smoothl1":
              mse_omics = [nn.SmoothL1Loss(reduction = "sum")(x, x_hat)/x.shape[1] for x, x_hat in zip(views, views_hat)]
            elif loss_type == "rmse":
              mse_omics = [torch.sqrt(nn.MSELoss(reduction = "sum")(x, x_hat)/x.shape[1]) for x, x_hat in zip(views, views_hat)]

            
            for i in range(len(mse_omics)):
                mse_list[f"Dataset {i}"].append(mse_omics[i].item() / n)
            
            loss_train.append(loss.item())
            mse_train.append(mse.item())
            kl_train.append(kl.item())
            
            with torch.autograd.set_detect_anomaly(True):
                # Backpropagate
                optimizer.zero_grad()
                loss.backward()
                #nn.utils.clip_grad_norm_(model.parameters(), max_norm=10, norm_type=2.0)
                optimizer.step()

        
        #--- VALIDATION LOOP
        model.eval()
        
        with torch.no_grad():
            for views in dataloader_test:
                views = [view.to(device) for view in views]
                # Forward pass to get the predictions
                views_hat = model.forward(views)

                # Get last layer of encoder with bottleneck
                h_bottleneck = model.encode(views)

                # Get means and log_vars
                means, log_variances = model.mean_variance(h_bottleneck)
                mu_joint, logvar_joint = model.product_of_experts(means, log_variances)

                # Calculate Losses
                loss, mse, kl = mse_kl(views_hat, views, mu_joint, logvar_joint, means, log_variances, loss_type = loss_type, alpha = alpha_KL, lambd = alpha_MSE)

                loss_val.append(loss.item())
                mse_val.append(mse.item())
                kl_val.append(kl.item())
            

    return loss_train, mse_train, kl_train, loss_val, mse_val, kl_val, mse_list


def objective(trial):
    # Generate the model.
    data = omics_db
    hidden_dim = trial.suggest_float("hidden_dim", 0.8, 1.0, log=False)
    latent_dim = trial.suggest_int("latent_dim", 90, 100)
    probability = 0.5
    group = 15
    activation_function = nn.Sigmoid()
    loss_type = "mse"
    alpha_KL = 0.01, 
    alpha_MSE = 0.99,
    n_folds = 3
    num_epochs = 15
    batch_size = trial.suggest_int("batch_size", 20, 100)

    model = OMIC_VAE(views = data.views, hidden_dim = hidden_dim, latent_dim = latent_dim, probability = probability, group = group, activation_function = activation_function).to(device)

    optimizer_name = trial.suggest_categorical("optimizer", ["Adam"])
    lr = trial.suggest_float("lr", 1e-5, 1e-4, log=True)
    w_decay = trial.suggest_float("w_decay", 1e-5, 1e-4, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr, weight_decay = w_decay)
    
    
    for epoch in range(num_epochs):
        
      #-- Cross Validation
      loss_train, mse_train, kl_train, loss_val, mse_val, kl_val, mse_list = cross_validation(
          data = data, n_folds = n_folds, 
          batch_size = batch_size, model = model, 
          optimizer = optimizer, alpha_KL = alpha_KL, 
          alpha_MSE = alpha_MSE,
          loss_type = loss_type)

      trial.report(np.mean(loss_val), epoch)

        # Handle pruning based on the intermediate value.
      if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    return np.mean(loss_val)


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    
    fig = optuna.visualization.plot_param_importances(study)
    dt = calendar.timegm(time.gmtime())
    fig.write_image(f'/home/sofiaapolinario/test/plots/optuna_best_param/{dt}best_param.pdf')





