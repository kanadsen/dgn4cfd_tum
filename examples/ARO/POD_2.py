import torch
import numpy as np
from torch.utils.data import DataLoader
import intel_extension_for_pytorch as ipex
import matplotlib.pyplot as plt

torch.multiprocessing.set_sharing_strategy('file_system')

#from dgn4cfd.datasets_3 import *
import sys
sys.path.insert(0,'/lus/flare/projects/Prob_AI/kanadsen/myrepos/dgn4cfd_tum/dgn4cfd')
from datasets_3 import *

print(torch.xpu.device_count())

def extract_pod_data(dataset, verbose=False):
    """
    Returns:
        params:  (M, 2)
        fields:  (M, N)
    """
    M = len(dataset)
    print("Printing POD data extraction...", M)

    # ---- constants ----
    mu_norm = 2e3
    p_shift = 2e3
    p_scale = 7e5

    # ---- infer field size from first sample ----
    first_graph = dataset[0]
    N = first_graph.target.numel()

    # ---- preallocate ----
    params = np.empty((M, 2), dtype=np.float32)
    fields = np.empty((M, N), dtype=np.float32)
    for i, graph in enumerate(dataset):
        # Parameters
        params[i, 0] = graph.glob[0, 0].item() / mu_norm
        params[i, 1] = graph.loc[0, 0].item() / mu_norm

        # Pressure field
        p = graph.target.view(-1)

        # Ensure CPU before numpy
        if p.is_cuda:
            p = p.cpu()

        fields[i] = (p.numpy() - p_shift) / p_scale

        if (i % 10 == 0 or i == M - 1):
            print(f"Extracted POD data {i+1}/{M}")

        if i==50:
            break

    print("End Printing POD data extraction...")
    return params, fields

def compute_pod(fields, energy_thresh=0.999):
    """
    fields: (M, N)
    """
    mean_field = fields.mean(axis=0)
    X = fields - mean_field

    # Snapshot POD
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    energy = np.cumsum(S**2) / np.sum(S**2)
    K = np.searchsorted(energy, energy_thresh) + 1

    modes = Vt[:K].T        # (N, K)
    coeffs = X @ modes     # (M, K)

    print(f"POD modes retained: {K}")
    print(f"Truncation error: {1 - energy[K-1]:.2e}")

    return mean_field, modes, coeffs

class PODDataset(torch.utils.data.Dataset):
    def __init__(self, params, coeffs):
        self.params = torch.tensor(params, dtype=torch.float32)
        self.coeffs = torch.tensor(coeffs, dtype=torch.float32)

    def __len__(self):
        return self.params.shape[0]

    def __getitem__(self, idx):
        return self.params[idx], self.coeffs[idx]

import torch.nn as nn

class PODMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=256, depth=4):
        super().__init__()

        layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, out_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def plot_error_histograms(field_errors, out_dir, epoch):
    # ---- Field error histogram ----
    plt.figure(figsize=(6, 4))
    plt.hist(field_errors, bins=15)
    plt.xlabel("Relative L2 Error (pressure field)")
    plt.ylabel("Count")
    plt.title("POD-MLP Test Error (Field Space)")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/pod_mlp_field_error_hist_{epoch}.png", dpi=300)
    plt.close()

def train_pod_mlp(model, loader, graph, params_test,fields_test,epochs=500, lr=1e-3, device="xpu"):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for ep in range(epochs):
        total = 0.0
        model.train()

        for p, a in loader:
            p, a = p.to(device), a.to(device)
            pred = model(p)
            loss = loss_fn(pred, a)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item()

        if ep % 5 == 0 and ep%50 !=0:
            print(f"Epoch {ep:04d} | Loss {total/len(loader):.4e}")
        elif ep %50 ==0:
            print(f"Epoch {ep:04d} | Loss {total/len(loader):.4e}")
            # Plot the results
            import random

            # Generate a random integer between 1 and 400
            idx = np.random.randint(0, len(params_test))

            p_pred = reconstruct_pressure(
                model,
                params_test[idx],
                mean_p,
                modes,
                device=device
            )
          
            p_pred = torch.tensor(p_pred, dtype=torch.float32).view(-1)
            original = (torch.tensor(fields_test[idx], dtype=torch.float32).view(-1)-2e3)/7e5
            
            field_errors = torch.abs(p_pred - original).numpy()

            plot_error_histograms(field_errors, out_dir="/lus/flare/projects/Prob_AI/kanadsen/myrepos/dgn4cfd_tum/examples/ARO/outputs_pressure", epoch=ep)
            graph.plot_pos_field(p_pred, azim=180, elev=0, s=0.05,filename=f"/lus/flare/projects/Prob_AI/kanadsen/myrepos/dgn4cfd_tum/examples/ARO/outputs_pressure/pressure_pod_mlp_pred_{ep}.png")
            graph.plot_pos_field(original, azim=180, elev=0, s=0.05,filename=f"/lus/flare/projects/Prob_AI/kanadsen/myrepos/dgn4cfd_tum/examples/ARO/outputs_pressure/pressure_original_{ep}.png")



def reconstruct_pressure(model, params, mean_field, modes, device="xpu"):
    model.eval()
    with torch.no_grad():
        mu = torch.tensor(params, dtype=torch.float32).to(device)
        a = model(mu).cpu().numpy()

    p = mean_field + modes @ a
    return p

def eval_pod_mlp(model, params_test, coeffs_test, device="xpu"):
    model.eval()
    errs = []

    with torch.no_grad():
        for i in range(len(params_test)):
            mu = torch.tensor(params_test[i], dtype=torch.float32).to(device)
            pred = model(mu).cpu().numpy()
            errs.append(np.linalg.norm(pred - coeffs_test[i]))

    print(f"Mean test coeff L2 error: {np.mean(errs):.3e}")

# -------------------------
# Training dataset
"""
transform = transforms.Compose([
    dgn.transforms.ConnectKNN(6),
    #dgn.transforms.ScaleEdgeAttr(0.015),                        # Scale the relative position stored as `edge_attr`
    #dgn.transforms.EdgeCondFreeStream(normals='loc'),           # Add the projection of the free stream velocity along edge-local axes as `edge_cond`
    dgn.transforms.ScaleAttr('target', vmin=2.5e+03,  vmax=1.5e+05),  # Scale the target field (pressure)
    dgn.transforms.ScaleAttr('glob', vmin=4,  vmax=5),  # Scale the target field (pressure)
    dgn.transforms.ScaleAttr('loc', vmin=203,  vmax=600),  # Scale the target field (pressure)
])
"""
# Create dataset
dataset = Shock(
    path="/lus/flare/projects/Prob_AI/kanadsen/all_data/ARO_data/dataset_trial2",
    T=1,
    transform = None,
    preload   = False,
)
graph = dataset.get_sequence(140, n_in=0)
print(graph.target.shape)
# Visualize target field
graph.plot_pos_field(graph.target.split(1, dim=1)[0].view(-1), azim=180, elev=0,s=0.05,filename=f"/lus/flare/projects/Prob_AI/kanadsen/myrepos/dgn4cfd_tum/examples/ARO/Test_image.png")

# Extract POD data
params, fields = extract_pod_data(dataset)

print("Params shape :", params.shape)
print("Fields shape :", fields.shape)


# POD
mean_p, modes, coeffs = compute_pod(fields, energy_thresh=0.999)

# -------------------------
# Train / Test split
# -------------------------
train_ratio = 0.95
num_samples = params.shape[0]

indices = np.random.permutation(num_samples)
train_size = int(train_ratio * num_samples)

train_idx = indices[:train_size]
test_idx  = indices[train_size:]

params_train = params[train_idx]
params_test  = params[test_idx]

coeffs_train = coeffs[train_idx]
coeffs_test  = coeffs[test_idx]

fields_train = fields[train_idx]
fields_test  = fields[test_idx]

print(f"Train samples: {len(train_idx)}")
print(f"Test samples : {len(test_idx)}")
pod_dataset_train = PODDataset(params_train, coeffs_train)
pod_dataset_test = PODDataset(params_test, coeffs_test)

train_loader = DataLoader(pod_dataset_train, batch_size=32, shuffle=True)


model = PODMLP(
    in_dim=params.shape[1],
    out_dim=coeffs.shape[1]
)

print("Training POD MLP...")
train_pod_mlp(model, train_loader, graph, params_test,
    fields_test, epochs=500)


eval_pod_mlp(model, params_test, coeffs_test)
