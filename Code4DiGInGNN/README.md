# An PyTorch implementation for our DiG-In-GNN
A Discriminative Feature Guided GNN against Inconsistency, to <i><u>dig in</u></i>to graphs for fraudsters. The code will be made public after the paper is accepted.

## Dependencies
<!-- `requirements.yaml` is for creating a conda environment.

Or it also can be installed step by step like: -->

```bash
# create a conda environment
conda create -n digin python=3.10.12
conda activate digin

# install pytorch
# If you prefer using your CPU, consider installing without cuda toolkit. Alternatively, you can install it without cuda toolkit, which may also work well on a CPU.
# conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 -c pytorch
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia

# sklearn
pip install -U scikit-learn

# DGL
# If you have installed dgl-cudaXX.X package, please uninstall it first.
conda install -c dglteam/label/cu117 dgl
# or if no cuda:
# conda install -c dglteam dgl

# other dependencies
pip install tqdm PyYAML matplotlib seaborn opencv-python
```

## Download the datasets

Due to the size limit of supplementary material, the dataset file `data.tar.gz` is uploaded to [Dropbox](https://www.dropbox.com/scl/fi/cas4ewe8s6o5nrvmlso14/data.tar.gz?rlkey=fwk7yaj3mz8o9icjq55mxdczc&dl=0) (`https://www.dropbox.com/scl/fi/cas4ewe8s6o5nrvmlso14/data.tar.gz?rlkey=fwk7yaj3mz8o9icjq55mxdczc&dl=0`) anonymously.

MD5: `c2623c886acb35d82661f05c4233d1c8`

The dataset directory should be like this tree:
```bash
.
└── data
    ├── amazon
    │   ├── Amazon.mat
    │   ├── amz_homo_adjlists.pickle
    │   ├── amz_upu_adjlists.pickle
    │   ├── amz_usu_adjlists.pickle
    │   └── amz_uvu_adjlists.pickle
    ├── tfinance
    │   ├── homo_tfinance.pickle
    │   └── tfinance
    └── yelpchi
        ├── YelpChi.mat
        ├── yelp_homo_adjlists.pickle
        ├── yelp_rsr_adjlists.pickle
        ├── yelp_rtr_adjlists.pickle
        └── yelp_rur_adjlists.pickle
```

## Device setting
By default, the device used is cuda:0. If you don't have a NVIDIA GPU, you can change the value of `cuda` to `False` in the `*.yaml` file. If you want to use another GPU or MPS, you can modify it in the `*.yaml` file like this:

```yaml
# this a snippet of amz.yaml
# Device
cuda: True
cuda_id: 2
device: 'cuda:2'
```
CPU:
```yaml
# this a snippet of amz.yaml
# Device
cuda: False
cuda_id: 0
device: 'cuda:0'
```

## Run the experiments
Now, everything seems ready. Let's run the experiments.

```bash
# YelpChi:
nohup python -u main.py --config ./config/yelp.yaml > logs/log_yelp.out 2>&1 &

# Amazon:
nohup python -u main.py --config ./config/amz.yaml > logs/log_amz.out 2>&1 &

# T-Finance
nohup python -u main.py --config ./config/tfin.yaml > log_tfin.out 2>&1 &
# Tip: since the number of edges in T-Finance is huge, using the config file `tfin_lowrate.yaml` in the `config` directory may be faster.
nohup python -u main.py --config ./config/tfin_lowrate.yaml > logs/log_tfin_lowrate.out 2>&1 &
```

Finally, we can take a cup of coffee now. You will see the results in the log files if everything goes well.