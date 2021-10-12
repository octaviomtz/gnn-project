import pandas as pd
import numpy as np
import torch

def weights_from_unbalanced_classes(df_name='data/raw/HIV_train.csv', target='HIV_active', debug_subset=False):
    '''return the weights of a unbalanced binary classes from a 
    column from a csv. From:
    https://discuss.pytorch.org/t/some-problems-with-weightedrandomsampler/23242/20'''
    df = pd.read_csv(df_name)
    classes = df[target].values
    if debug_subset:
        classes = [i for idx, i in enumerate(classes) if idx % 5 == 0]
    class0 = np.sum(classes==0)
    class1 = np.sum(classes==1)
    weights = 1/torch.Tensor(np.asarray([class0,class1]))
    samples_weight = torch.tensor([weights[t] for t in classes])
    return weights, samples_weight

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
