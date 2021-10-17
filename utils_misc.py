import pandas as pd
import numpy as np
import torch
from sklearn.metrics import (confusion_matrix, f1_score, accuracy_score, 
                            precision_score, recall_score, roc_auc_score)
import mlflow.pytorch
from tqdm import tqdm 

def weights_from_unbalanced_classes(df_name='data/raw/HIV_train.csv', target='HIV_active', debug_subset=False):
    '''return the weights of a unbalanced binary classes from a 
    column from a csv. From:
    https://discuss.pytorch.org/t/some-problems-with-weightedrandomsampler/23242/20'''
    df = pd.read_csv(df_name)
    classes = df[target].values
    # if debug_subset:
    #     classes = [i for idx, i in enumerate(classes) if idx % 5 == 0]
    class0 = np.sum(classes==0)
    class1 = np.sum(classes==1)
    weights = 1/torch.Tensor(np.asarray([class0,class1]))
    samples_weight = torch.tensor([weights[t] for t in classes])
    return weights, samples_weight

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def calculate_metrics(y_pred, y_true, epoch, type):
    print(f"\n Confusion matrix: \n {confusion_matrix(y_pred, y_true)}")
    print(f"F1 Score: {f1_score(y_pred, y_true):.03f}")
    print(f"Accuracy: {accuracy_score(y_pred, y_true):.03f}")
    print(f"Precision: {precision_score(y_pred, y_true):.03f}")
    print(f"Recall: {recall_score(y_pred, y_true):.03f}")
    try:
        roc = roc_auc_score(y_pred, y_true)
        print(f"ROC AUC: {roc:.03f}")
        mlflow.log_metric(key=f"ROC-AUC-{type}", value=float(roc), step=epoch)
    except:
        mlflow.log_metric(key=f"ROC-AUC-{type}", value=float(0), step=epoch)
        print(f"ROC AUC: notdefined")

def train(model, loader, loss_fn, optimizer, device, epoch):
    # Enumerate over the data
    all_preds = []
    all_labels = []
    for _, batch in enumerate(tqdm(loader, desc='Train')):
        batch.to(device)  
        optimizer.zero_grad() 
        pred = model(batch.x.float(), 
                                batch.edge_attr.float(),
                                batch.edge_index, 
                                batch.batch) 
        loss = loss_fn(pred, batch.y.float())
        loss.backward()  
        optimizer.step()  
        #this was working with CrossEntropyLoss but not with BCEWithLogitsLoss
        # all_preds.append(np.argmax(pred.cpu().detach().numpy(), axis=1))
        all_preds.append(np.rint(torch.sigmoid(pred).cpu().detach().numpy()))
        all_labels.append(batch.y.cpu().detach().numpy())
    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()
    calculate_metrics(all_preds, all_labels, epoch, "train")
    return loss

def test(model, loader, loss_fn, optimizer, device, epoch):
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(loader, desc='Test'):
            batch.to(device)  
            pred = model(batch.x.float(), 
                            batch.edge_attr.float(),
                            batch.edge_index, 
                            batch.batch) 
            loss = loss_fn(pred, batch.y.float())
            all_preds.append(np.rint(torch.sigmoid(pred).cpu().detach().numpy()))  
            # all_preds.append(np.argmax(pred.cpu().detach().numpy(), axis=1))
            all_labels.append(batch.y.cpu().detach().numpy())
        
    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()
    calculate_metrics(all_preds, all_labels, epoch, "test")
    return loss