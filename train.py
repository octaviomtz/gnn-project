#%% imports 
import torch 
from torch_geometric.loader import DataLoader
import numpy as np
from tqdm import tqdm
from dataset_featurizer import MoleculeDataset
# from model import GNN
from models import GNN, GNNTrans
import mlflow.pytorch
from load_from_folder import ProcessedDataset
from getpass import getpass
from utils_misc import (weights_from_unbalanced_classes, count_parameters,
                        train, test)
import os
from torch.utils.data import WeightedRandomSampler
import hydra
from omegaconf import DictConfig, OmegaConf
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from config import BEST_PARAMETERS as params

# TODO
# 1.  Use BCEWithLogitsLoss for GNN too.
# 1a. The ouput should be 1 neuron 
# 1b. Adapt to use the running loss 

@hydra.main(config_path='config', config_name='config.yaml')
def main(cfg:DictConfig):
    path_orig = hydra.utils.get_original_cwd()
    # Specify tracking server
    os.environ['MLFLOW_TRACKING_USERNAME'] = cfg.dags_user
    os.environ['MLFLOW_TRACKING_PASSWORD'] = getpass('Enter your DAGsHub access token: ')
    mlflow.set_tracking_uri(cfg.dags_uri)

    # Call this only on processed after 
    ## Loading the dataset
    # train_dataset = MoleculeDataset(root="data/", filename="HIV_train_oversampled.csv")
    # test_dataset = MoleculeDataset(root="data/", filename="HIV_test.csv")
    test_dataset = ProcessedDataset(f'{path_orig}/data/processed_test', debug_subset=cfg.debug_subset)
    train_dataset = ProcessedDataset(f'{path_orig}/data/processed_train', debug_subset=cfg.debug_subset)

    # Load one model
    if cfg.model == 'GNNTrans':
        params["model_edge_dim"] = train_dataset[0].edge_attr.shape[1]
        # print(params["model_edge_dim"], params["model_embedding_size"], params["model_attention_heads"], params["model_dropout_rate"], params["model_edge_dim"], params['model_layers'])
        model = GNNTrans(feature_size=train_dataset[0].x.shape[1], model_params=params) 
        model_name='GNN transformer'
        # Loss and Optimizer
        weight = torch.tensor([params["pos_weight"]], dtype=torch.float32).to(device)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=weight)
    else: #default to GNN
        model = GNN(feature_size=train_dataset[0].x.shape[1]) 
        model_name='GNN'
        # Loss and Optimizer
        if cfg.weighted_loss:
            weights = torch.tensor([1, 10], dtype=torch.float32).to(device)
            loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
        else:
            loss_fn = torch.nn.CrossEntropyLoss()
    model = model.to(device)
    print(f"{cfg.model} model: {model_name} Number of parameters: {count_parameters(model)}")

    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)  
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    # Prepare training
    if cfg.weighted_sampler:
        weights, samples_weight = weights_from_unbalanced_classes(df_name=f'{path_orig}/data/raw/HIV_train.csv', target='HIV_active', debug_subset=cfg.debug_subset)
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, 
                            num_workers=2, sampler=sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, 
                        num_workers=2, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, 
                            num_workers=2, shuffle=False)


    # Run the training
    with mlflow.start_run() as run:
        for epoch in range(cfg.epochs): # 500
            # Training
            model.train()
            loss = train(model, train_loader, loss_fn, optimizer, device, epoch=epoch)
            loss = loss.detach().cpu().numpy()
            print(f"Epoch {epoch} | Train Loss {loss}")
            mlflow.log_metric(key="Train loss", value=float(loss), step=epoch)

            # Testing
            model.eval()
            if epoch % cfg.test_ever_epoch == 0:
                loss = test(model, test_loader, loss_fn, optimizer, device, epoch=epoch)
                loss = loss.detach().cpu().numpy()
                print(f"Epoch {epoch} | Test Loss {loss}")
                mlflow.log_metric(key="Test loss", value=float(loss), step=epoch)
        
            scheduler.step()
        print("Done.")

    if cfg.save_model:
        torch.save(model, f'{path_orig}models/model_GNN_.pt')

if __name__ == '__main__':
    main()

# %% Save the model 
# mlflow.pytorch.log_model(model, "model")


