import numpy as np
import torch
from torch.utils.data import DataLoader

from data import simulate_lorenz, PairDataset
from models import DeepKoopmanCT


def set_seed(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)


def rollout(model: DeepKoopmanCT, x0, dt, steps):
    """Autoregressive rollout: x_{t+1} = decoder(expm(A dt) encoder(x_t))"""
    xs = [x0]
    x = x0
    for _ in range(steps):
        out = model.forward_one_step(x.unsqueeze(0), dt)
        x = out["x_tp1_pred"].squeeze(0)
        xs.append(x)
    return torch.stack(xs, dim=0)


def train_deep_koopman_ct(
    autoencoder,
    device="cuda" if torch.cuda.is_available() else "cpu",
    dt=0.01,
    n_traj=50,
    n_steps=5000,
    burn_in=200,
    batch_size=2048,
    epochs=50,
    lr=1e-3,
    latent_dim=8,
    alpha_latent=1.0,
    beta_recon=1.0,
    gamma_predx=1.0,
    grad_clip=1.0,
    seed=0,
    traj=None,
):
    """Train a DeepKoopmanCT model on trajectory data.

    Parameters
    ----------
    autoencoder : torch.nn.Module
        An autoencoder instance (MLPAutoEncoder or KANAutoEncoder).
        Must have .encoder and .decoder attributes and return (encoded, reconstructed).
    traj : torch.Tensor, optional
        Pre-generated trajectory data of shape (n_traj, n_steps, state_dim).
        If None, simulates Lorenz trajectories using the other parameters.
    """
    set_seed(seed)

    if traj is None:
        traj = simulate_lorenz(
            n_traj=n_traj, n_steps=n_steps, dt=dt, burn_in=burn_in, device=device
        )
    else:
        traj = traj.to(device)

    n_train = int(0.8 * n_traj)
    train_traj = traj[:n_train].contiguous()
    val_traj = traj[n_train:].contiguous()

    train_ds = PairDataset(train_traj)
    val_ds = PairDataset(val_traj)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    ae = autoencoder.to(device)
    model = DeepKoopmanCT(ae, latent_dim=latent_dim).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    mse = torch.nn.MSELoss()

    history = {
        "train_total": [], "train_predx": [], "train_latent": [], "train_recon": [],
        "val_total": [], "val_predx": [], "val_latent": [], "val_recon": [],
    }

    def eval_loop():
        model.eval()
        totals, predxs, latents, recons = [], [], [], []
        with torch.no_grad():
            for x_t, x_tp1 in val_dl:
                x_t = x_t.to(device)
                x_tp1 = x_tp1.to(device)

                out = model.forward_one_step(x_t, dt)
                z_tp1_true = model.ae.encoder(x_tp1)

                loss_predx = mse(out["x_tp1_pred"], x_tp1)
                loss_latent = mse(out["z_tp1_pred"], z_tp1_true)
                loss_recon = mse(out["x_t_rec"], x_t)
                loss_total = gamma_predx * loss_predx + alpha_latent * loss_latent + beta_recon * loss_recon

                totals.append(loss_total.item())
                predxs.append(loss_predx.item())
                latents.append(loss_latent.item())
                recons.append(loss_recon.item())

        return (
            float(np.mean(totals)),
            float(np.mean(predxs)),
            float(np.mean(latents)),
            float(np.mean(recons)),
        )

    for ep in range(1, epochs + 1):
        model.train()
        totals, predxs, latents, recons = [], [], [], []

        for x_t, x_tp1 in train_dl:
            x_t = x_t.to(device)
            x_tp1 = x_tp1.to(device)

            out = model.forward_one_step(x_t, dt)
            z_tp1_true = model.ae.encoder(x_tp1)

            loss_predx = mse(out["x_tp1_pred"], x_tp1)
            loss_latent = mse(out["z_tp1_pred"], z_tp1_true)
            loss_recon = mse(out["x_t_rec"], x_t)
            loss_total = gamma_predx * loss_predx + alpha_latent * loss_latent + beta_recon * loss_recon

            opt.zero_grad(set_to_none=True)
            loss_total.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

            totals.append(loss_total.item())
            predxs.append(loss_predx.item())
            latents.append(loss_latent.item())
            recons.append(loss_recon.item())

        val_total, val_predx, val_latent, val_recon = eval_loop()

        history["train_total"].append(float(np.mean(totals)))
        history["train_predx"].append(float(np.mean(predxs)))
        history["train_latent"].append(float(np.mean(latents)))
        history["train_recon"].append(float(np.mean(recons)))
        history["val_total"].append(val_total)
        history["val_predx"].append(val_predx)
        history["val_latent"].append(val_latent)
        history["val_recon"].append(val_recon)

        if ep == 1 or ep % 5 == 0 or ep == epochs:
            print(
                f"Epoch {ep:03d} | "
                f"train total {history['train_total'][-1]:.5f} (predx {history['train_predx'][-1]:.5f}, "
                f"latent {history['train_latent'][-1]:.5f}, recon {history['train_recon'][-1]:.5f}) | "
                f"val total {val_total:.5f}"
            )

    return model, history, train_traj.detach().cpu(), val_traj.detach().cpu(), dt
