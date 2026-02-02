import torch
import numpy as np
from torch.utils.data import DataLoader
import intel_extension_for_pytorch as ipex
import matplotlib.pyplot as plt

torch.multiprocessing.set_sharing_strategy('file_system')

study = "pressure_norm"

#from dgn4cfd.datasets_3 import *
import sys
sys.path.insert(0,'/lus/flare/projects/Prob_AI/kanadsen/myrepos/dgn4cfd_tum/dgn4cfd')
from datasets_3 import *

print(torch.xpu.device_count())

def extract_pod_data(dataset, verbose=True):
    """
    Returns:
        params : (M_valid, 3)
        fields : (M_valid, N)
    """
    print("Starting POD data extraction...")
    print(f"Total dataset size: {len(dataset)}")

    # ---- constants ----
    mu_norm = 2e3

    params_list = []
    fields_list = []

    # ---- infer field size ----
    first_graph = dataset[0]
    N = first_graph.target.numel()

    for i, graph in enumerate(dataset):

        # ---- parameters (normalized) ----
        parameters_norm = torch.tensor([
            graph.glob[0, 0].item() / mu_norm,
            graph.loc[0, 0].item() / mu_norm,
            graph.loc[0, 1].item() / 5.0
        ], dtype=torch.float32)

        # ---- skip near-zero regimes ----
        if torch.all(parameters_norm < 0.1):
            continue

        # ---- pressure field ----
        p = (graph.target.detach().cpu().view(-1) - 2e3) / 7e5  # normalize pressure field

        params_list.append(parameters_norm.numpy())
        fields_list.append(p.numpy())

        if verbose and (len(params_list) % 10 == 0):
            print(f"Valid samples collected: {len(params_list)}")

        # optional hard cap (now works correctly)
        #if len(params_list) == 50:
        #    break

    params = np.stack(params_list, axis=0)
    fields = np.stack(fields_list, axis=0)

    print(f"Finished POD data extraction")
    print(f"Valid samples retained: {params.shape[0]}")
    print(f"Field dimension: {fields.shape[1]}")

    return params, fields


def plot_modes(modes, graph, S):
    """
    Plots the POD modes using the provided graph for geometry.
    """
    K = modes.shape[1]

    if S is not None:
        energy = (S**2) / np.sum(S**2)

    for k in range(K):
        mode_k = torch.from_numpy(modes[:, k]).float().view(-1)

        title = None
        if S is not None:
            title = f"Mode {k+1} | Energy = {energy[k]:.2e}"

        graph.plot_pos_field_2D(
            mode_k,
            azim=180,
            elev=0,
            s=0.1,
            title=title,
            filename=f"/lus/flare/projects/Prob_AI/kanadsen/myrepos/dgn4cfd_tum/examples/ARO/outputs_{study}/{title}.png"
        )

def plot_pod_energy(S, energy_thresh=None, savepath=f"/lus/flare/projects/Prob_AI/kanadsen/myrepos/dgn4cfd_tum/examples/ARO/outputs_{study}/mode_energy.png"):
    energy = S**2
    energy /= energy.sum()
    cum_energy = np.cumsum(energy)

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.bar(
        np.arange(1, len(energy)+1),
        energy,
        alpha=0.6,
        label="Energy per mode"
    )

    ax.plot(
        np.arange(1, len(cum_energy)+1),
        cum_energy,
        "o-",
        color="black",
        label="Cumulative energy"
    )

    if energy_thresh is not None:
        ax.axhline(
            energy_thresh,
            linestyle="--",
            color="red",
            label=f"Threshold = {energy_thresh}"
        )

    ax.set_xlabel("Mode index")
    ax.set_ylabel("Normalized energy")
    ax.set_title("POD energy spectrum")
    ax.legend()
    ax.grid(True)

    if savepath:
        fig.savefig(savepath, bbox_inches="tight")

    plt.show()

def compute_pod(fields, graph, energy_thresh=0.999):
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
    # Plot the energy spectrum
    plot_pod_energy(S)

    # Plot the modes
    plot_modes(modes, graph, S)

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
    def __init__(self, in_dim, out_dim, hidden_dim=256, depth=8):
        super().__init__()

        layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, out_dim))

        self.net = nn.Sequential(*layers)

        # apply initialization
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

def plot_error_histograms(field_errors, out_dir, epoch):
    # ---- Field error histogram ----
    plt.figure(figsize=(6, 4))
    plt.hist(field_errors, bins=15)
    plt.xlabel(f"Relative L2 Error {study}")
    plt.ylabel("Count")
    plt.title("POD-MLP Test Error (Field Space)")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/pod_mlp_field_error_hist_{epoch}.png", dpi=300)
    plt.close()

def train_pod_mlp(model, loader, graph, params_test,fields_test,epochs=500, lr=5e-4, device="xpu"):
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
            model.eval()

            # Generate a random integer between 1 and 400
            idx = np.random.randint(0, len(params_test))

            p_pred = reconstruct_pressure(
                model,
                params_test[idx],
                mean_p,
                modes,
                device=device
            )
          
            p_pred = torch.tensor(p_pred, dtype=torch.float32).view(-1)*7e5 + 2e3  # denormalize pressure field
            original = (torch.tensor(fields_test[idx], dtype=torch.float32).view(-1))*7e5 + 2e3  # denormalize pressure field
            
            field_errors = torch.abs(p_pred - original).numpy()

            #plot_error_histograms(field_errors, out_dir=f"/lus/flare/projects/Prob_AI/kanadsen/myrepos/dgn4cfd_tum/examples/ARO/outputs_{study}", epoch=ep)
            #graph.plot_pos_field_2D(torch.abs(p_pred - original), title=f"Error for {params_test[idx]}", azim=180, elev=0, s=0.05,filename=f"/lus/flare/projects/Prob_AI/kanadsen/myrepos/dgn4cfd_tum/examples/ARO/outputs_{study}/mach_pod_mlp_error_{ep}.png")
            #graph.plot_pos_field_2D(p_pred, title=f"Prediction for {params_test[idx]}", azim=180, elev=0, s=0.1,filename=f"/lus/flare/projects/Prob_AI/kanadsen/myrepos/dgn4cfd_tum/examples/ARO/outputs_{study}/mach_pod_mlp_pred_{ep}.png")
            #graph.plot_pos_field_2D(original, title=f"Original for {params_test[idx]}", azim=180, elev=0, s=0.1,filename=f"/lus/flare/projects/Prob_AI/kanadsen/myrepos/dgn4cfd_tum/examples/ARO/outputs_{study}/mach_pod_original_{ep}.png")
            model.train()



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


def save_dataset_frames(
    model,
    params,
    fields,
    mean_p,
    modes,
    graph,
    epoch,
    out_dir,
    tag,
    device="xpu",
):
    """
    tag: 'train' or 'test'
    """
    model.eval()

    frame_dir_pred = os.path.join(out_dir, "frames_true")
    os.makedirs(frame_dir_pred, exist_ok=True)

    frame_dir_true = os.path.join(out_dir, "frames_pred")
    os.makedirs(frame_dir_true, exist_ok=True)

    frame_dir_error = os.path.join(out_dir, "frames_error")
    os.makedirs(frame_dir_error, exist_ok=True)

    frame_dir_all = os.path.join(out_dir, "frames_all")
    os.makedirs(frame_dir_all, exist_ok=True)

    for i in range(len(params)):
        p_pred = reconstruct_pressure(
            model,
            params[i],
            mean_p,
            modes,
            device=device,
        )

        p_pred = torch.tensor(p_pred, dtype=torch.float32).view(-1) * 7e5 + 2e3
        original = torch.tensor(fields[i], dtype=torch.float32).view(-1) * 7e5 + 2e3

        error = torch.abs(p_pred - original)

        fname_true = os.path.join(frame_dir_true, f"{i:04d}.png")
        fname_pred = os.path.join(frame_dir_pred, f"{i:04d}.png")
        fname_error = os.path.join(frame_dir_error, f"{i:04d}.png")
        fname_all = os.path.join(frame_dir_all, f"{i:04d}.png")

        """
        graph.plot_pos_field_2D(
            error,
            title=f"Prediction for {params[i]}",
            azim=180,
            elev=0,
            s=0.05,
            filename=fname_error,
        )

        graph.plot_pos_field_2D(
            p_pred,
            title=f"Prediction for {params[i]}",
            azim=180,
            elev=0,
            s=0.05,
            filename=fname_pred,
        )

        graph.plot_pos_field_2D(
            original,
            title=f"Prediction for {params[i]}",
            azim=180,
            elev=0,
            s=0.05,
            filename=fname_true,
        )
        """

        graph.plot_pos_field_uvw_2D(
            u = p_pred,
            v = original,
            w = error,
            title=f"Prediction for {params[i]}",
            s=0.1,
            vmin_uv=2e3,
            vmax_uv=6e5,
            vmin_w=0,
            vmax_w=12000,
            filename=fname_all,
        )

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
graph.plot_pos_field_2D(graph.target.split(1, dim=1)[0].view(-1), azim=180, elev=0,s=0.05,filename=f"/lus/flare/projects/Prob_AI/kanadsen/myrepos/dgn4cfd_tum/examples/ARO/Test_image.png")

# Extract POD data
params, fields = extract_pod_data(dataset)

print("Params shape :", params.shape)
print("Fields shape :", fields.shape)


# POD
mean_p, modes, coeffs = compute_pod(fields, graph, energy_thresh=0.999)

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

save_dataset_frames(
    model,
    params_train,
    fields_train,
    mean_p,
    modes,
    graph,
    epoch="final",
    out_dir=f"/lus/flare/projects/Prob_AI/kanadsen/myrepos/dgn4cfd_tum/examples/ARO/outputs_{study}/outputs_train",
    tag="train",
)

save_dataset_frames(
    model,
    params_test,
    fields_test,
    mean_p,
    modes,
    graph,
    epoch="final",
    out_dir=f"/lus/flare/projects/Prob_AI/kanadsen/myrepos/dgn4cfd_tum/examples/ARO/outputs_{study}/outputs_test",
    tag="test",
)