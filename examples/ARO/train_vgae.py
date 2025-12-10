"""
    Train a Variational Graph Autoencoder (VGAE) to predict the pressure field on an ellipse.
    Run with:
        python train_vgae.py --experiment_id 0 --gpu 0
        
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

experiment = {
    0: {
        'name':                 'VGAE_model',
        'width':                64,
        'latent_node_features': 4,
        'kl_reg':               0.001,
        'depths':               [2,2,2],
        'width':                64,
        'nt':                   1, # Limit the length of the training simulations to 10 timesteps
    },
}[args.experiment_id]

# Training settings
train_settings = dgn.nn.TrainingSettings(
    name             = experiment['name'],
    folder           = './checkpoints',
    checkpoint       = './checkpoints/{experiment["name"]}.chk',  
    tensor_board     = './boards',
    chk_interval     = 1,
    training_loss    = dgn.nn.losses.VaeLoss(kl_reg=experiment['kl_reg']),
    epochs           = 1000,
    batch_size       = 3,
    lr               = 1e-4,
    grad_clip        = {"epoch": 0, "limit": 1},
    scheduler        = {"factor": 0.1, "patience": 50, "loss": "training"},
    stopping         = 1e-8,
    device           = torch.device(f'cuda:{args.gpu}') if args.gpu >= 0 else torch.device('cpu'),
)

# Training dataset
# Training dataset
transform = transforms.Compose([
    dgn.transforms.ScaleAttr('target', vmin=2.5e+03,  vmax=1.5e+05),  # Scale the target field (pressure)
    dgn.transforms.ScaleAttr('glob', vmin=4,  vmax=5),  # Scale the target field (pressure)
    dgn.transforms.ScaleAttr('loc', vmin=203,  vmax=600),  # Scale the target field (pressure)
    #dgn.transforms.ScaleAttr('param_1', vmin=2700,  vmax=3600),  # Scale the target field (pressure)
    dgn.transforms.MeshCoarsening(                              # Create 5 lower-resolution graphs and normalise the relative position betwen the inter-graph nodes.
        num_scales      = 3,
        rel_pos_scaling = [0.015, 0.03, 0.06], #  
        scalar_rel_pos  = True, 
    ),
    dgn.transforms.Copy('target', 'field'), # Because the target is the input field
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
    'dim':                  3, # 3D 
    'in_node_features':     1, # p
    'cond_node_features':   2, # Re, d_bottom, d_top
    'cond_edge_features':   3, # x_j - x_i, y_j - y_i, U_\inf projection
    'latent_node_features': experiment['latent_node_features'],
    'depths':               experiment['depths'],
    'fnns_depth':           2,
    'fnns_width':           experiment['width'],
    'aggr':                'sum',
    'dropout':             0.1,
}
model = dgn.nn.VGAE(arch = arch)

# Training
model.fit(train_settings, dataloader)