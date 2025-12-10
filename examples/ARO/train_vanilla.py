"""
    Train a determinisitic GNN to predict the pressure field on an ellipse.
    Run with:
        python train_vanilla.py --experiment_id 0 --gpu 0
"""

import torch
from torchvision import transforms
import argparse

import dgn4cfd as dgn

torch.multiprocessing.set_sharing_strategy('file_system')
from dgn4cfd.datasets_3 import *


argparser = argparse.ArgumentParser()
argparser.add_argument('--experiment_id', type=int)
argparser.add_argument('--gpu',  type=int, default=0)
args = argparser.parse_args()

# Initial seed
seed = 0
torch.manual_seed(seed)

# Dictionary of experiments
experiment = {
    0: {
        'name':     'Vanilla_GNN',
        'depths':   [2,2,2],
        'width':    100,
        'nt':        1, # Limit the length of the training simulations to 10 timesteps
    },
}[args.experiment_id]

# Training settings
train_settings = dgn.nn.TrainingSettings(
    name             = experiment['name'],
    folder           = './checkpoints',
    checkpoint       = f'./checkpoints/{experiment["name"]}.chk',
    tensor_board     = './boards',
    chk_interval     = 1,
    training_loss    = dgn.nn.MseLoss(),
    epochs           = 1000,
    batch_size       = 4,
    lr               = 0.001,
    grad_clip        = {"epoch": 0, "limit": 1},
    scheduler        = {"factor": 0.1, "patience": 50, "loss": 'training'},
    stopping         = 1e-8,
    device           = torch.device(f'cuda:{args.gpu}') if args.gpu >= 0 else torch.device('cpu'),
)

# Training dataset
transform = transforms.Compose([
    dgn.transforms.ConnectKNN(6),
    dgn.transforms.ScaleAttr('target', vmin=2.5e+03,  vmax=1.5e+05),  # Scale the target field (pressure)
    dgn.transforms.ScaleAttr('glob', vmin=4,  vmax=5),  # Scale the target field (pressure)
    dgn.transforms.ScaleAttr('loc', vmin=203,  vmax=600),  # Scale the target field (pressure)
    #dgn.transforms.ScaleAttr('param_1', vmin=2700,  vmax=3600),  # Scale the target field (pressure)
    dgn.transforms.MeshCoarsening(                              # Create 3 lower-resolution graphs and normalise the relative position betwen the inter-graph nodes.
        num_scales      =  3,
        rel_pos_scaling = [0.02, 0.06, 0.15],
        scalar_rel_pos  = True, 
    ),
])
dataset = Shock(
    path      = "/home/kanadsen01/Desktop/Git_repos/Forked_Repos/dgn4cfd_tum/data/dataset_trial1",
    T         = experiment['nt'],
    transform = transform,
    preload   = False,
)
dataloader = dgn.DataLoader(
    dataset     = dataset,
    batch_size  = train_settings['batch_size'],
    shuffle     = True,
    num_workers = 4,    
)   

# Model
arch = {
    'dim': 3,
    'in_node_features':   0,
    'cond_node_features': 2, # Re, d_bottom, d_top
    'cond_edge_features': 3, # x_j - x_i, y_j - y_i, U_\inf projection
    'out_node_features':  1, # Pressure mean
    'depths':             experiment['depths'],
    'fnns_width':         experiment['width'],
    'aggr':               'sum',
    'dropout':            0.1,
}
model = dgn.nn.VanillaGnn(arch = arch)

# Training
model.fit(train_settings, dataloader)