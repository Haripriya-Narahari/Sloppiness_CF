import torch._dynamo
torch._dynamo.config.suppress_errors = True

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse.linalg import LinearOperator, eigsh
from collections import OrderedDict
import gc

# "Better" functional tools + vmap
from torch.func import functional_call, jvp, grad, vmap

# --- Optimization: Enable TF32 ---
torch.set_float32_matmul_precision('high')

# --- Configuration ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
REP_DIM = 4
TASK_A_DIGIT = 1
TASK_B_DIGIT = 8
CENTER_1 = None
CENTER_2 = None

torch.manual_seed(42)
np.random.seed(42)

# --- Dataset & Model ---
class SingleClassMNIST(torch.utils.data.Dataset):
    def __init__(self, base_dataset, digit):
        self.x = []
        for img, label in base_dataset:
            if label == digit:
                self.x.append(img)
        self.x = torch.stack(self.x)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], 0

class SVDDNetwork(nn.Module):
    def __init__(self, hidden_sizes=[10,8], out_dim=4):
        super().__init__()
        layers = []
        in_size = 28 * 28
        for h in hidden_sizes:
            layers += [nn.Linear(in_size, h), nn.ReLU()]
            in_size = h
        layers.append(nn.Linear(in_size, out_dim, bias=False))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))

def svdd_loss(outputs, center):
    return torch.mean(torch.sum((outputs - center) ** 2, dim=1))

def get_fixed_center(dim, device):
    c = torch.randn(dim).to(device)
    return (c / torch.norm(c)).detach()

def get_param_vec(model):
    return torch.cat([p.view(-1) for p in model.parameters()])

def set_param_vec(model, param_vec):
    idx = 0
    for p in model.parameters():
        numel = p.numel()
        p.data = param_vec[idx:idx + numel].view_as(p).clone()
        idx += numel

def evaluate_dist(model, loader, center, device):
    model.eval()
    total_dist, count = 0, 0
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            out = model(x)
            total_dist += torch.norm(out - center, dim=1).sum().item()
            count += x.size(0)
    return total_dist / count

# --- Step 1: Train Digit 1 + Hessian computation (FIXED) ---
def train_digit_1(k_max=128):  # Reduced for speed
    print(f"Phase 1: Learning Digit {TASK_A_DIGIT}")
    global CENTER_1
    CENTER_1 = get_fixed_center(REP_DIM, DEVICE)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_raw = MNIST(root='./data', train=True, download=True, transform=transform)
    test_raw = MNIST(root='./data', train=False, download=True, transform=transform)

    loader_a_train = DataLoader(SingleClassMNIST(train_raw, TASK_A_DIGIT), batch_size=1024, shuffle=True)
    loader_a_test = DataLoader(SingleClassMNIST(test_raw, TASK_A_DIGIT), batch_size=1000)

    model = SVDDNetwork(out_dim=REP_DIM).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # Train Phase 1 (Adam is fine here)
    for epoch in range(10):
        model.train()
        for x, _ in loader_a_train:
            x = x.to(DEVICE)
            opt.zero_grad()
            out = model(x)
            loss = svdd_loss(out, CENTER_1)
            loss.backward()
            opt.step()

    dist_1 = evaluate_dist(model, loader_a_test, CENTER_1, DEVICE)
    print(f"Baseline Dist: {dist_1:.4f}")

    # --- FIXED HVP Setup ---
    theta_star_dict = model.state_dict()
    params_at_theta = OrderedDict()
    for k, v in theta_star_dict.items():
        params_at_theta[k] = v.detach().to(DEVICE).clone().requires_grad_(True)

    param_items = list(params_at_theta.items())
    param_keys = [k for k, _ in param_items]
    param_shapes = [v.shape for _, v in param_items]
    param_numels = [v.numel() for _, v in param_items]
    slices = []
    start = 0
    for n in param_numels:
        slices.append(slice(start, start + n))
        start += n
    num_params = start

    def functional_loss(params, data, c):
        out = functional_call(model, params, (data,))
        return svdd_loss(out, c)

    print("Compiling gradient function...")
    grad_func = grad(functional_loss)
    fast_grad_func = torch.compile(grad_func, mode="reduce-overhead")

    def compute_batch_jvp(tangents_dict, params_dict, data, c):
        zeros_data = torch.zeros_like(data)
        zeros_c = torch.zeros_like(c)
        return jvp(fast_grad_func, (params_dict, data, c), (tangents_dict, zeros_data, zeros_c))[1]

    batched_jvp_op = vmap(compute_batch_jvp, in_dims=(0, None, None, None))

    # Multi-batch for stable Hessian approx
    print("Loading 4 batches for Hessian approximation...")
    subset_batches = []
    iter_loader = iter(loader_a_train)
    try:
        for _ in range(4):
            subset_batches.append(next(iter_loader)[0].to(DEVICE))
    except StopIteration:
        pass
    print(f"Using {len(subset_batches)} batches for HVP.")

    def hvp_block_wrapper(V_np):
        if V_np.ndim == 1:
            V_np = V_np[:, None]

        V_tensor = torch.from_numpy(V_np).to(DEVICE).float().T
        current_block_size = V_tensor.shape[0]

        v_tensors = []
        for s, shp in zip(slices, param_shapes):
            sub = V_tensor[:, s]
            v_tensors.append(sub.view(current_block_size, *shp))

        v_dict_batch = OrderedDict((k, t) for k, t in zip(param_keys, v_tensors))

        acc_hvp = None
        # FIXED: NO torch.no_grad() around JVP!
        for x_batch in subset_batches:
            out_dict = batched_jvp_op(v_dict_batch, params_at_theta, x_batch, CENTER_1)
            out_flat = torch.cat([out_dict[k].reshape(current_block_size, -1) for k in param_keys], dim=1)

            if acc_hvp is None:
                acc_hvp = out_flat
            else:
                acc_hvp += out_flat

        avg_hvp = acc_hvp / float(len(subset_batches))
        return avg_hvp.detach().cpu().numpy().astype(np.float32)  # detach only at end

    def hvp_matvec_wrapper(v_np):
        return hvp_block_wrapper(v_np).flatten()

    print(f"Computing Hessian (Top {k_max}) using eigsh...")

    lin_op = LinearOperator(
        (num_params, num_params),
        matvec=hvp_matvec_wrapper,
        matmat=hvp_block_wrapper,
        dtype=np.float32
    )

    safe_k = min(k_max, num_params - 1)

    # FIXED: which='LA' + positive filter
    vals, vecs = eigsh(
        lin_op,
        k=safe_k,
        which='LA',  # Largest ALGEBRAIC (prioritizes positive)
        v0=np.random.normal(size=(num_params,))
    )

    # Sort descending algebraic
    idx = np.argsort(vals)[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]

    # FIXED: Keep ONLY positive curvature
    pos_mask = vals > 1e-8
    if not np.any(pos_mask):
        raise RuntimeError("No positive Hessian eigenvalues found!")
    pos_vals = vals[pos_mask]
    pos_vecs = vecs[:, pos_mask]

    print(f"Top positive eigenvalue: {pos_vals[0]:.4f} (found {len(pos_vals)} positive evals)")

    def pick_stiff_dim(evals, frac=0.9, min_dim=1, max_dim=None):
        lam = np.maximum(evals, 0.0)
        total = lam.sum()
        if total <= 0:
            return min_dim
        cumsum = np.cumsum(lam)
        target = frac * total
        d = int(np.searchsorted(cumsum, target) + 1)
        if max_dim is not None:
            d = min(d, max_dim)
        d = max(d, min_dim)
        return d

    STIFF_FRAC = 0.9
    stiff_dim = pick_stiff_dim(pos_vals, frac=STIFF_FRAC, max_dim=len(pos_vals))
    print(f"Picked stiff_dim = {stiff_dim} out of {len(pos_vals)} positive")

    stiff_subspace_full = torch.from_numpy(pos_vecs.copy()).float().to(DEVICE).T  # (k_pos, num_params)

    return {
        'model_state': theta_star_dict,
        'theta_star': params_at_theta,
        'stiff_subspace_full': stiff_subspace_full,
        'stiff_dim': stiff_dim,
        'loader_a_test': loader_a_test
    }

def flatten_params_dict(param_dict):
    return torch.cat([p.view(-1) for p in param_dict.values()])


def intrinsic_projected_sgd(model, lr, theta_star_vec, basis, mode):
    # 1. Get grad at CURRENT point
    g = torch.cat([p.grad.view(-1) for p in model.parameters()])

    # 2. Project grad (tangent space is same everywhere for linear subspace)
    if mode == "stiff":
        g_proj = basis.T @ (basis @ g)
    elif mode == "sloppy":
        g_stiff = basis.T @ (basis @ g)
        g_proj = g - g_stiff
    else:
        g_proj = g

    # 3. Step from CURRENT point
    current_theta = get_param_vec(model)
    new_theta = current_theta - lr * g_proj

    # 4. (Optional but good) Re-anchor to ensure no drift
    # The new point should be: theta_star + (something in subspace)
    # Since we only stepped in subspace directions, this is automatically satisfied
    # up to floating point error.

    set_param_vec(model, new_theta)


# --- Helper: Reusable Hessian Computation ---
def compute_hessian_basis(model, loader, center, k_dim, device):
    """
    Re-computes the top-k eigenvectors of the Hessian at the CURRENT model parameters.
    """
    # 1. Setup Functional Call
    params_dict = dict(model.named_parameters())
    keys = list(params_dict.keys())
    values = [params_dict[k] for k in keys]

    # Pre-process params for functional call
    # We need to detach them to treat them as constants for the HVP setup
    params_at_theta = {k: v.detach().clone().requires_grad_(True) for k, v in params_dict.items()}

    # Shapes for reshaping flattened vectors
    param_shapes = [v.shape for v in values]
    param_numels = [v.numel() for v in values]
    slices = []
    start = 0
    for n in param_numels:
        slices.append(slice(start, start + n))
        start += n
    num_params = start

    # 2. Define Gradient & JVP (Same as before)
    def functional_loss(p_dict, data, c):
        out = functional_call(model, p_dict, (data,))
        return svdd_loss(out, c)

    grad_func = grad(functional_loss)
    # Note: re-compiling every time might be slow, but safe for periodic updates
    # For max speed, you'd cache the compiled function, but let's keep it simple.

    def compute_batch_jvp(tangents_dict, params_dict, data, c):
        zeros_data = torch.zeros_like(data)
        zeros_c = torch.zeros_like(c)
        return jvp(grad_func, (params_dict, data, c), (tangents_dict, zeros_data, zeros_c))[1]

    batched_jvp_op = vmap(compute_batch_jvp, in_dims=(0, None, None, None))

    # 3. Get Batch Data (Just grab a few batches)
    subset_batches = []
    iter_loader = iter(loader)
    try:
        for _ in range(4): # 4 Batches
            subset_batches.append(next(iter_loader)[0].to(device))
    except StopIteration:
        pass

    # 4. Define Linear Operator
    def hvp_block_wrapper(V_np):
        if V_np.ndim == 1: V_np = V_np[:, None]
        V_tensor = torch.from_numpy(V_np).to(device).float().T
        current_block_size = V_tensor.shape[0]

        # Unflatten to dict
        v_tensors = []
        for s, shp in zip(slices, param_shapes):
            sub = V_tensor[:, s]
            v_tensors.append(sub.view(current_block_size, *shp))
        v_dict_batch = OrderedDict((k, t) for k, t in zip(keys, v_tensors))

        acc_hvp = None
        for x_batch in subset_batches:
            # JVP requires tracking, detach only at end
            out_dict = batched_jvp_op(v_dict_batch, params_at_theta, x_batch, center)
            out_flat = torch.cat([out_dict[k].reshape(current_block_size, -1) for k in keys], dim=1)
            if acc_hvp is None: acc_hvp = out_flat
            else: acc_hvp += out_flat

        return (acc_hvp / len(subset_batches)).detach().cpu().numpy().astype(np.float32)

    def hvp_matvec(v): return hvp_block_wrapper(v).flatten()

    lin_op = LinearOperator((num_params, num_params), matvec=hvp_matvec, matmat=hvp_block_wrapper, dtype=np.float32)

    # 5. Eigsh
    k_safe = min(k_dim, num_params - 1)
    vals, vecs = eigsh(lin_op, k=k_safe, which='LA', v0=np.random.normal(size=(num_params,)))

    # Filter Positive
    pos_mask = vals > 1e-8
    if not np.any(pos_mask): return None
    pos_vecs = vecs[:, pos_mask]

    # Return Transposed Basis (k, num_params)
    return torch.from_numpy(pos_vecs.copy()).float().to(device).T

# --- Modified Train Function ---
def train_digit_2(experiment_data, k, constraint_type, recompute_freq=5, task1_loader=None):
    """
    recompute_freq: How many epochs to wait before updating the Stiff Subspace.
                    Set to 0 or None to use static (original behavior).
    task1_loader:   Needed to calculate Hessian for Task 1.
    """
    global CENTER_2
    CENTER_2 = -CENTER_1.clone()

    theta_star_dict = experiment_data['theta_star']
    # Initial Basis from Step 1
    stiff_subspace_full = experiment_data['stiff_subspace_full']
    theta_star_vec = flatten_params_dict(theta_star_dict).detach()

    # Load Model
    model = SVDDNetwork(out_dim=REP_DIM).to(DEVICE)
    model.load_state_dict(experiment_data['model_state'])

    # Setup Initial Constraint
    k_eff = min(k, stiff_subspace_full.shape[0])
    constraint_basis = stiff_subspace_full[:k_eff] if constraint_type != "none" else None

    print(f"[{constraint_type}] Start Basis: {constraint_basis.shape if constraint_basis is not None else 'None'}")

    # Data Loaders (Task B)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_raw = MNIST(root='./data', train=True, download=True, transform=transform)
    test_raw = MNIST(root='./data', train=False, download=True, transform=transform)
    loader_b_train = DataLoader(SingleClassMNIST(train_raw, TASK_B_DIGIT), batch_size=1024, shuffle=True)
    loader_b_test = DataLoader(SingleClassMNIST(test_raw, TASK_B_DIGIT), batch_size=1000)

    lr = 0.01
    angles = []

    for epoch in range(20):
        model.train()
        for x, _ in loader_b_train:
            x = x.to(DEVICE)
            out = model(x)
            loss = svdd_loss(out, CENTER_2)

            model.zero_grad()
            loss.backward()

            if constraint_type == "none":
                with torch.no_grad():
                    for p in model.parameters():
                        p -= lr * p.grad
            else:
                # 2. Project using the STATIC constraint_basis
                intrinsic_projected_sgd(model, lr, theta_star_vec, constraint_basis, constraint_type)

                # 3. Log Angle (This is now perfectly accurate!)
                with torch.no_grad():
                    delta = get_param_vec(model) - theta_star_vec
                    if torch.norm(delta) > 1e-9:
                        proj_stiff = (delta @ constraint_basis.T) @ constraint_basis
                        cosine = torch.norm(proj_stiff) / torch.norm(delta)
                        cosine = torch.clamp(cosine, 0.0, 1.0)
                        angle_deg = torch.acos(cosine) * 57.2958
                        angles.append(angle_deg.item())

    dist_1 = evaluate_dist(model, experiment_data['loader_a_test'], CENTER_1, DEVICE)
    dist_2 = evaluate_dist(model, loader_b_test, CENTER_2, DEVICE)
    avg_angle = np.mean(angles) if angles else 0.0

    return {'k': k, 'scenario': constraint_type, 'dist_1': dist_1, 'dist_2': dist_2, 'avg_angle': avg_angle}


if __name__ == '__main__':
    data = train_digit_1(k_max=1000)  # Fast
    print("\nPhase 2 (Conflict Mode: Opposite Centers)...")
    results = []

    for k in [2, 5, 10, 25, 50, 100, 250, 500, 1000]:  # Match available subspace
        print(f"--- k={k} ---")
        for sc in ["sloppy", "stiff", "none"]:
            res = train_digit_2(data, k, sc)
            results.append(res)
            ang_str = f" | Angle w/ Stiff: {res['avg_angle']:.1f}°" if sc != 'none' else ""
            print(f"[{sc}] Dist 1: {res['dist_1']:.4f} | Dist 2: {res['dist_2']:.4f} {ang_str}")

    # Plot
    df = pd.DataFrame(results)
    df.to_csv('results_fixed.csv', index=False)
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 6))

    for sc in ['sloppy', 'stiff', 'none']:
        subset = df[df['scenario'] == sc]
        if sc == 'none':
            ax1.axhline(y=subset['dist_1'].iloc[0], color='gray', linestyle='--', label='Unconstrained')
            ax2.axhline(y=subset['dist_2'].iloc[0], color='gray', linestyle='--', label='Unconstrained')
        else:
            ax1.plot(subset['k'], subset['dist_1'], marker='o', label=sc.title())
            ax2.plot(subset['k'], subset['dist_2'], marker='o', label=sc.title())

    ax1.set_title(f"Retention of Digit {TASK_A_DIGIT} (Lower Better)", fontsize=16)
    ax1.set_xlabel("k (Stiff Dimensions)",fontsize=14)
    ax1.legend(fontsize=14)
    ax1.set_ylabel("Distance to CENTER_1",fontsize=14)

    ax2.set_title(f"Learning Digit {TASK_B_DIGIT} (Lower Better)", fontsize=16)
    ax2.set_xlabel("k (Stiff Dimensions)",fontsize=14)
    ax2.legend(fontsize=14)
    ax2.set_ylabel("Distance to CENTER_2",fontsize=14)

    for sc in ['sloppy', 'stiff']:
        subset = df[df['scenario'] == sc]
        ax3.plot(subset['k'], subset['avg_angle'], marker='s', linestyle='-', label=sc.title())
    ax3.set_title("Update Angle vs Stiff Subspace",fontsize=16)
    ax3.set_ylabel("Degrees (90°=Sloppy, 0°=Stiff)",fontsize=14)
    ax3.set_xlabel("k (Stiff Dimensions)",fontsize=14)
    ax3.legend(fontsize=14)

    plt.tight_layout()
    plt.savefig('one_class_eigsh_fixed.png', dpi=1000, bbox_inches='tight')
    plt.show()
    print("\nSaved 'one_class_eigsh_fixed.png' and 'results_fixed.csv'")
