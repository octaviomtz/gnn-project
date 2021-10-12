#%% imports 
import torch 
from torch_geometric.loader import DataLoader
from sklearn.metrics import confusion_matrix, f1_score, \
    accuracy_score, precision_score, recall_score, roc_auc_score
import numpy as np
from tqdm import tqdm
from dataset_featurizer import MoleculeDataset
from model import GNN
from model_old import GNN
import mlflow.pytorch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from load_from_folder import ProcessedDataset
from getpass import getpass
from utils_misc import weights_from_unbalanced_classes, count_parameters
import os
from torch.utils.data import WeightedRandomSampler

#%% Specify tracking server
# mlflow.set_tracking_uri("http://localhost:5000")
os.environ['MLFLOW_TRACKING_USERNAME'] = 'octaviomtz'
os.environ['MLFLOW_TRACKING_PASSWORD'] = getpass('Enter your DAGsHub access token: ')
mlflow.set_tracking_uri(f'https://dagshub.com/octaviomtz/gnn-project.mlflow')

#%% Call this only on processed after 
## Loading the dataset
# train_dataset = MoleculeDataset(root="data/", filename="HIV_train_oversampled.csv")
# test_dataset = MoleculeDataset(root="data/", filename="HIV_test.csv")
debug_subset=False
test_dataset = ProcessedDataset('data/processed_test', debug_subset=debug_subset)
train_dataset = ProcessedDataset('data/processed_train', debug_subset=debug_subset)

#%% Loading the model
model = GNN(feature_size=train_dataset[0].x.shape[1]) 
model = model.to(device)
print(f"Number of parameters: {count_parameters(model)}")
model

#%% Loss and Optimizer
weights = torch.tensor([1, 10], dtype=torch.float32).to(device)
loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)  
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)


#%% Prepare training
BATCH_SIZE = 256
weights, samples_weight = weights_from_unbalanced_classes(df_name='data/raw/HIV_train.csv', target='HIV_active', debug_subset=debug_subset)
sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
train_loader = DataLoader(train_dataset, 
                    batch_size=BATCH_SIZE, 
                    num_workers=2, sampler=sampler)#shuffle=True)
test_loader = DataLoader(test_dataset, 
                         batch_size=BATCH_SIZE, 
                         num_workers=2, shuffle=False)

def train(epoch):
    # Enumerate over the data
    all_preds = []
    all_labels = []
    for _, batch in enumerate(tqdm(train_loader)):
        batch.to(device)  
        optimizer.zero_grad() 
        pred = model(batch.x.float(), 
                                batch.edge_attr.float(),
                                batch.edge_index, 
                                batch.batch) 
        loss = torch.sqrt(loss_fn(pred, batch.y)) 
        loss.backward()  
        optimizer.step()  

        all_preds.append(np.argmax(pred.cpu().detach().numpy(), axis=1))
        all_labels.append(batch.y.cpu().detach().numpy())
    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()
    calculate_metrics(all_preds, all_labels, epoch, "train")
    return loss

def test(epoch):
    all_preds = []
    all_labels = []
    for batch in test_loader:
        batch.to(device)  
        pred = model(batch.x.float(), 
                        batch.edge_attr.float(),
                        batch.edge_index, 
                        batch.batch) 
        loss = torch.sqrt(loss_fn(pred, batch.y))    
        all_preds.append(np.argmax(pred.cpu().detach().numpy(), axis=1))
        all_labels.append(batch.y.cpu().detach().numpy())
    
    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()
    calculate_metrics(all_preds, all_labels, epoch, "test")
    return loss


def calculate_metrics(y_pred, y_true, epoch, type):
    print(f"\n Confusion matrix: \n {confusion_matrix(y_pred, y_true)}")
    print(f"F1 Score: {f1_score(y_pred, y_true)}")
    print(f"Accuracy: {accuracy_score(y_pred, y_true)}")
    print(f"Precision: {precision_score(y_pred, y_true)}")
    print(f"Recall: {recall_score(y_pred, y_true)}")
    try:
        roc = roc_auc_score(y_pred, y_true)
        print(f"ROC AUC: {roc}")
        mlflow.log_metric(key=f"ROC-AUC-{type}", value=float(roc), step=epoch)
    except:
        mlflow.log_metric(key=f"ROC-AUC-{type}", value=float(0), step=epoch)
        print(f"ROC AUC: notdefined")

# %% Run the training
with mlflow.start_run() as run:
    for epoch in range(11): # 500
        # Training
        model.train()
        loss = train(epoch=epoch)
        loss = loss.detach().cpu().numpy()
        print(f"Epoch {epoch} | Train Loss {loss}")
        mlflow.log_metric(key="Train loss", value=float(loss), step=epoch)

        # Testing
        model.eval()
        if epoch % 5 == 0:
            loss = test(epoch=epoch)
            loss = loss.detach().cpu().numpy()
            print(f"Epoch {epoch} | Test Loss {loss}")
            mlflow.log_metric(key="Test loss", value=float(loss), step=epoch)
    
        scheduler.step()
    print("Done.")


# %% Save the model 
mlflow.pytorch.log_model(model, "model")

#%%
weights
# %%
weights[train_targets]
# %%
import pandas as pd
df = pd.read_csv('data/raw/HIV_train.csv')
classes = df['HIV_active'].values
# %%
classes
# %%
samples_weight = torch.tensor([weights[t] for t in classes])
samples_weight
# %%
