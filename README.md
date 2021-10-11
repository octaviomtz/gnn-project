# Installing requirments to use colab via ssh
```bash
pip3 install -r requirements.txt
bash install_geometric.sh
```

# Comments about the code
This is the code for this video series: https://www.youtube.com/watch?v=nAEb1lOf_4o

## Further things
- Its highly recommended to setup a GPU (including CUDA) for this code. 
- Here is where I found ideas for node / edge features: https://www.researchgate.net/figure/Descriptions-of-node-and-edge-features_tbl1_339424976
- There is also a Kaggle competition that used this dataset (from a University):
https://www.kaggle.com/c/iml2019/overview

## Dashboard (MLFlow + Streamlit)
It is required to use conda for this setup, e.g.
```
wget https://repo.continuum.io/archive/Anaconda3-5.3.1-Linux-x86_64.sh
``` 

You need to start the following things:
- Streamlit server
```
streamlit run dashboard.py
```

- MlFlow Server
```
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./artifacts \
    --host 0.0.0.0
    --port 5000
```

- MlFlow served model
```
export MLFLOW_TRACKING_URI=http://localhost:5000
mlflow models serve -m "models:/YourModelName/Staging" -p 1234
```

TODO: Check if multi-input models work for MLFLOW!!!
