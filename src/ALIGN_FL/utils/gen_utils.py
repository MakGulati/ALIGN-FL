import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import inception_v3
from torchmetrics.image.inception import InceptionScore

# Set Matplotlib backend to avoid GUI warnings
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from scipy.linalg import sqrtm

import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from sklearn.neural_network import MLPClassifier
from torchvision.utils import save_image
from sklearn.decomposition import PCA
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend to avoid thread issues
from matplotlib import pyplot as plt
import random
import os
from prv_accountant import PRVAccountant
from prv_accountant.privacy_random_variables import PoissonSubsampledGaussianMechanism
from sklearn.mixture import GaussianMixture
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
from sklearn.exceptions import UndefinedMetricWarning

# Suppress the specific warnings
warnings.filterwarnings(
    "ignore", category=FutureWarning, module="sklearn.linear_model._logistic"
)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

warnings.filterwarnings(
    "ignore",
    message="Metric `InceptionScore` will save all extracted features in buffer.*",
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


class VAE(nn.Module):

    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder with more layers, larger hidden dimensions, and additional regularization
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        # Decoder with more layers, larger hidden dimensions, and additional regularization
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 512),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1024),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, input_dim),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class SimpleVAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(SimpleVAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Simplified encoder with ReLU activation
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            # nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            # nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            # nn.Dropout(0.3),
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

        # Simplified decoder with ReLU activation
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            # nn.Dropout(0.3),
            nn.Linear(128, 256),
            nn.ReLU(),
            # nn.Dropout(0.3),
            nn.Linear(256, 512),
            nn.ReLU(),
            # nn.Dropout(0.3),
            nn.Linear(512, input_dim),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def loss_function(
    recon_x,
    x,
    mu,
    logvar,
    lambda_kl=1,
    normalize=False,
):
    if recon_x.size() != x.size():
        x = x.view(recon_x.size())
    # BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction="sum")
    # if normalize:
    #     BCE = BCE / x.size(0)
    # KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # if normalize:
    #     KLD = KLD / x.size(0)
    MSE = F.mse_loss(recon_x, x, reduction="none").sum(dim=1)
    MSE = MSE.mean()
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD * lambda_kl
    # return BCE + KLD * lambda_kl


def mse_loss_function_with_gp(
    model,
    data,
    recon_batch,
    mu,
    log_var,
    BETA=1.0,
    use_gp=True,
    gradient_penalty_weight=0.1,
    penalty_at="decoder",
):
    # Normalize reconstruction loss by batch size
    MSE = F.mse_loss(recon_batch, data.view(-1, 784), reduction="none").sum(dim=1)
    MSE = MSE.mean()

    # Compute KLD loss

    KLD = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

    # Beta-VAE formulation with stronger KLD weight
    # Increased KLD weight to constrain latent space
    loss = MSE + BETA * KLD

    if use_gp and gradient_penalty_weight > 0:
        z = model.reparameterize(mu, log_var)

        # Compute GP on both real latents and random samples
        if penalty_at == "decoder":
            print("Computing GP at decoder")
            gp_real = compute_lipschitz_penalty(model, z)
        else:
            print("Computing GP at encoder")
            gp_real = compute_lipschitz_penalty_encoder(model, data)

        gp = gp_real

        loss = loss + gradient_penalty_weight * gp
    return loss, MSE, KLD, gp if (use_gp and gradient_penalty_weight > 0) else None


class SampleDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def compute_lipschitz_penalty(model, z, epsilon=1e-7):

    # Sample random points between latent samples
    alpha = torch.rand(z.size(0), 1).to(z.device)
    interpolated_z = alpha * z + (1 - alpha) * z[torch.randperm(z.size(0))]
    interpolated_z.requires_grad_(True)

    with torch.enable_grad():
        # Get decoder outputs for interpolated points
        decoded = model.decode(interpolated_z)

        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=decoded,
            inputs=interpolated_z,
            grad_outputs=torch.ones_like(decoded),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        # Calculate Lipschitz penalty
        gradients = gradients.view(gradients.size(0), -1)
        gradients_norm = torch.sqrt(torch.sum(gradients**2, dim=1) + epsilon)
        # lipschitz_penalty = torch.mean((torch.relu(gradients_norm - threshold)) ** 2)
        lipschitz_penalty = torch.mean((gradients_norm - 1) ** 2)

    return lipschitz_penalty


def compute_lipschitz_penalty_encoder(model, x, epsilon=1e-7):

    # Sample random points between input samples
    x = x.view(-1, 784)
    alpha = torch.rand(x.size(0), 1).to(x.device)
    interpolated_x = alpha * x + (1 - alpha) * x[torch.randperm(x.size(0))]
    interpolated_x.requires_grad_(True)

    with torch.enable_grad():

        # Get encoder outputs for interpolated points
        mu, logvar = model.encode(interpolated_x)
        gradients_mu = torch.autograd.grad(
            outputs=mu,
            inputs=interpolated_x,
            grad_outputs=torch.ones_like(mu),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients_logvar = torch.autograd.grad(
            outputs=logvar,
            inputs=interpolated_x,
            grad_outputs=torch.ones_like(logvar),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients_mu + gradients_logvar

        # Calculate Lipschitz penalty
        gradients = gradients.view(gradients.size(0), -1)
        gradients_norm = torch.sqrt(torch.sum(gradients**2, dim=1) + epsilon)
        lipschitz_penalty = torch.mean((gradients_norm - 1) ** 2)

    return lipschitz_penalty


def compute_decoder_lipschitz(model):
    """Compute the theoretical Lipschitz constant of the decoder"""
    lip_constant = 1.0
    for module in model.decoder:
        if isinstance(module, nn.Linear):
            # Get the actual spectral norm
            weight = module.weight.data
            spectral_norm = torch.linalg.matrix_norm(weight, ord=2)
            lip_constant *= spectral_norm.item()
        elif isinstance(module, nn.LeakyReLU):
            lip_constant *= max(1, module.negative_slope)
        elif isinstance(module, nn.Sigmoid):
            lip_constant *= 0.25
        elif isinstance(module, nn.ReLU):
            lip_constant *= 1
    return lip_constant


def compute_epoch_lipschitz_constant(model, test_loader, device, mode="decoder"):
    """
    Compute Lipschitz constant using test data for current epoch.
    """
    model.eval()
    max_gradient_norm = 0

    with torch.no_grad():  # Initial forward pass to get latents
        all_z = []
        all_x = []
        for data, _ in test_loader:
            data = data.to(device)
            if mode == "decoder":
                # Get latent vectors
                mu, log_var = model.encode(data.view(-1, 784))
                z = model.reparameterize(mu, log_var)
                all_z.append(z)
            else:
                all_x.append(data.view(-1, 784))

    if mode == "decoder":
        all_points = torch.cat(all_z, dim=0)
    else:
        all_points = torch.cat(all_x, dim=0)

    # Compute gradients in batches
    batch_size = 128
    for i in range(0, len(all_points), batch_size):
        points = all_points[i : i + batch_size]
        points.requires_grad_(True)

        with torch.enable_grad():
            if mode == "decoder":
                output = model.decode(points)
            else:
                output, _, _ = model(points)

            gradients = torch.autograd.grad(
                outputs=output,
                inputs=points,
                grad_outputs=torch.ones_like(output),
                create_graph=False,
                retain_graph=False,
                only_inputs=True,
            )[0]

        gradient_norms = torch.norm(gradients.view(gradients.size(0), -1), dim=1)
        max_batch_norm = gradient_norms.max().item()
        max_gradient_norm = max(max_gradient_norm, max_batch_norm)

    return max_gradient_norm


def estimate_empirical_lipschitz(model, z1, z2=None):
    """Estimate empirical Lipschitz constant using random samples"""
    if z2 is None:
        # Generate random perturbations
        z2 = z1 + torch.randn_like(z1) * 0.1

    with torch.no_grad():
        out1 = model.decode(z1)
        out2 = model.decode(z2)

        input_diffs = torch.norm(z1 - z2, dim=1)
        output_diffs = torch.norm(out1 - out2, dim=1)

        # Compute Lipschitz quotients
        lip_quotients = output_diffs / (input_diffs + 1e-8)
        return torch.max(lip_quotients).item()


def collect_latent_statistics(model, local_dataloader, device):
    """
    Collect statistics for mu and log_var across the dataset
    """
    model.eval()
    all_mu = []
    all_log_var = []

    with torch.no_grad():
        for batch_idx, data in enumerate(local_dataloader):
            features = data[0].to(device)
            _, mu, log_var = model(features)
            all_mu.append(mu.cpu().numpy())
            all_log_var.append(log_var.cpu().numpy())

    all_mu = np.concatenate(all_mu, axis=0)
    all_log_var = np.concatenate(all_log_var, axis=0)

    # Calculate statistics for each dimension
    mu_stats = []
    log_var_stats = []

    for dim in range(all_mu.shape[1]):
        mu_dim = all_mu[:, dim]
        log_var_dim = all_log_var[:, dim]

        # Calculate statistics for mu
        mu_stats.append(
            {
                "min": np.min(mu_dim),
                "q1": np.percentile(mu_dim, 25),
                "median": np.median(mu_dim),
                "q3": np.percentile(mu_dim, 75),
                "max": np.max(mu_dim),
                "mean": np.mean(mu_dim),
            }
        )

        # Calculate statistics for log_var
        log_var_stats.append(
            {
                "min": np.min(log_var_dim),
                "q1": np.percentile(log_var_dim, 25),
                "median": np.median(log_var_dim),
                "q3": np.percentile(log_var_dim, 75),
                "max": np.max(log_var_dim),
                "mean": np.mean(log_var_dim),
            }
        )

    return {"mu": {"dimensions": mu_stats}, "log_var": {"dimensions": log_var_stats}}


def local_model_train_only(local_trainset, model, config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    all_latent_vectors = []

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = config["local_epochs"]
    lambda_kl = config["lambda_kl"]
    lambda_lip = config["lambda_lip"]
    latent_dim = config["latent_dim"]
    folder = config["folder"]
    server_round = config["server_round"]
    cid = config["cid"]
    local_dataloader = DataLoader(
        local_trainset, batch_size=config["batch_size"], shuffle=True
    )

    for epoch in range(num_epochs):
        total_train_loss = total_lip_loss = 0
        total_mse = 0
        total_kl_loss = 0
        total_gp_loss = 0
        epoch_lip_constants = []

        for batch_idx, data in enumerate(local_dataloader):
            features = data[0].to(device)
            optimizer.zero_grad()

            recon_batch, mu, log_var = model(features)
            z = model.reparameterize(mu, log_var)
            all_latent_vectors.append(z.detach().cpu().numpy())

            vae_loss, mse, kl, gp_loss = mse_loss_function_with_gp(
                model,
                features,
                recon_batch,
                mu,
                log_var,
                lambda_kl,
                use_gp=True,
                gradient_penalty_weight=lambda_lip,
            )

            vae_loss.backward()
            optimizer.step()
            total_train_loss += vae_loss.item()
            total_mse += mse.item()
            total_kl_loss += kl.item()
            total_gp_loss += gp_loss.item() if lambda_lip > 0 else 0
    # Fit GMM on collected latent vectors
    all_latent_vectors = np.concatenate(all_latent_vectors, axis=0)
    gmm = GaussianMixture(n_components=10, covariance_type="diag", random_state=42)
    gmm.fit(all_latent_vectors)
    save_gmm(gmm, f"{folder}/gmm_c_{cid}_r{server_round}.joblib")
    all_loss = {
        "local_total_train_loss": total_train_loss / len(local_dataloader),
        "local_total_mse": total_mse / len(local_dataloader),
        "local_total_kl_loss": total_kl_loss / len(local_dataloader),
        "local_total_gp_loss": total_gp_loss / len(local_dataloader),
    }
    # Collect latent space statistics
    # latent_stats = collect_latent_statistics(model, local_dataloader, device)

    # Save statistics
    # np.save(f"{folder}/latent_stats_c_{cid}_r{server_round}.npy", latent_stats)
    return all_loss, model


def local_model_train_only_enc_penalty(local_trainset, model, config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    all_latent_vectors = []

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = config["local_epochs"]
    lambda_kl = config["lambda_kl"]
    lambda_lip = config["lambda_lip"]
    latent_dim = config["latent_dim"]
    folder = config["folder"]
    server_round = config["server_round"]
    cid = config["cid"]
    local_dataloader = DataLoader(
        local_trainset, batch_size=config["batch_size"], shuffle=True
    )

    for epoch in range(num_epochs):
        total_train_loss = total_lip_loss = 0
        total_mse = 0
        total_kl_loss = 0
        total_gp_loss = 0
        epoch_lip_constants = []

        for batch_idx, data in enumerate(local_dataloader):
            features = data[0].to(device)
            optimizer.zero_grad()

            recon_batch, mu, log_var = model(features)
            z = model.reparameterize(mu, log_var)
            all_latent_vectors.append(z.detach().cpu().numpy())

            vae_loss, mse, kl, gp_loss = mse_loss_function_with_gp(
                model,
                features,
                recon_batch,
                mu,
                log_var,
                lambda_kl,
                use_gp=True,
                gradient_penalty_weight=lambda_lip,
                penalty_at="encoder",
            )

            vae_loss.backward()
            optimizer.step()
            total_train_loss += vae_loss.item()
            total_mse += mse.item()
            total_kl_loss += kl.item()
            total_gp_loss += gp_loss.item() if lambda_lip > 0 else 0
    # Fit GMM on collected latent vectors
    all_latent_vectors = np.concatenate(all_latent_vectors, axis=0)
    gmm = GaussianMixture(n_components=10, covariance_type="diag", random_state=42)
    gmm.fit(all_latent_vectors)
    save_gmm(gmm, f"{folder}/gmm_c_{cid}_r{server_round}.joblib")
    all_loss = {
        "local_total_train_loss": total_train_loss / len(local_dataloader),
        "local_total_mse": total_mse / len(local_dataloader),
        "local_total_kl_loss": total_kl_loss / len(local_dataloader),
        "local_total_gp_loss": total_gp_loss / len(local_dataloader),
    }
    # Collect latent space statistics
    # latent_stats = collect_latent_statistics(model, local_dataloader, device)

    # Save statistics
    # np.save(f"{folder}/latent_stats_c_{cid}_r{server_round}.npy", latent_stats)
    return all_loss, model


def standard_local_model_train(local_trainset, model, config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    all_latent_vectors = []

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = config["local_epochs"]
    lambda_kl = config["lambda_kl"]
    lambda_lip = config["lambda_lip"]
    latent_dim = config["latent_dim"]
    folder = config["folder"]
    server_round = config["server_round"]
    cid = config["cid"]
    local_dataloader = DataLoader(
        local_trainset, batch_size=config["batch_size"], shuffle=True
    )

    for epoch in range(num_epochs):
        total_train_loss = total_lip_loss = 0
        total_mse = 0
        total_kl_loss = 0
        total_gp_loss = 0
        epoch_lip_constants = []

        for batch_idx, data in enumerate(local_dataloader):
            features = data[0].to(device)
            optimizer.zero_grad()

            recon_batch, mu, log_var = model(features)
            z = model.reparameterize(mu, log_var)
            all_latent_vectors.append(z.detach().cpu().numpy())

            vae_loss, mse, kl, gp_loss = mse_loss_function_with_gp(
                model,
                features,
                recon_batch,
                mu,
                log_var,
                lambda_kl,
                use_gp=True,
                gradient_penalty_weight=lambda_lip,
            )

            vae_loss.backward()
            optimizer.step()
            total_train_loss += vae_loss.item()
            total_mse += mse.item()
            total_kl_loss += kl.item()
            total_gp_loss += gp_loss.item() if lambda_lip > 0 else 0

    all_loss = {
        "local_total_train_loss": total_train_loss / len(local_dataloader),
        "local_total_mse": total_mse / len(local_dataloader),
        "local_total_kl_loss": total_kl_loss / len(local_dataloader),
        "local_total_gp_loss": total_gp_loss / len(local_dataloader),
    }

    return all_loss, model


def standard_local_model_train_prox(local_trainset, model, config, global_model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    all_latent_vectors = []

    global_params = [
        val.detach().cpu().numpy() for _, val in global_model.state_dict().items()
    ]

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = config["local_epochs"]
    lambda_kl = config["lambda_kl"]
    lambda_lip = config["lambda_lip"]
    proximal_mu = config.get(
        "proximal_mu", 0.01
    )  # proximal term coefficient, default 0.01

    local_dataloader = DataLoader(
        local_trainset, batch_size=config["batch_size"], shuffle=True
    )

    for epoch in range(num_epochs):
        total_train_loss = total_lip_loss = 0
        total_mse = total_kl_loss = total_gp_loss = total_prox_loss = 0

        for batch_idx, data in enumerate(local_dataloader):
            features = data[0].to(device)
            optimizer.zero_grad()

            recon_batch, mu, log_var = model(features)
            z = model.reparameterize(mu, log_var)
            all_latent_vectors.append(z.detach().cpu().numpy())

            vae_loss, mse, kl, gp_loss = mse_loss_function_with_gp(
                model,
                features,
                recon_batch,
                mu,
                log_var,
                lambda_kl,
                use_gp=True,
                gradient_penalty_weight=lambda_lip,
            )

            # Calculate proximal term
            proximal_term = 0.0
            model_params = [
                val.detach().cpu().numpy() for _, val in model.state_dict().items()
            ]
            for local_param, global_param in zip(model_params, global_params):
                diff = local_param - global_param
                # proximal_term += (diff**2).sum()
                proximal_term += np.linalg.norm(diff, ord=2) ** 2

            proximal_term = (proximal_mu / 2) * proximal_term
            print(f"Proximal term: {proximal_term.item()}, Loss: {vae_loss.item()}")

            # Add proximal term to total loss
            total_loss = vae_loss + proximal_term

            total_loss.backward()
            optimizer.step()

            # Track all losses
            total_train_loss += total_loss.item()
            total_mse += mse.item()
            total_kl_loss += kl.item()
            total_gp_loss += gp_loss.item() if lambda_lip > 0 else 0
            total_prox_loss += proximal_term.item()

    # Average losses
    num_batches = len(local_dataloader)
    all_loss = {
        "local_total_train_loss": total_train_loss / num_batches,
        "local_total_mse": total_mse / num_batches,
        "local_total_kl_loss": total_kl_loss / num_batches,
        "local_total_gp_loss": total_gp_loss / num_batches,
        "local_total_prox_loss": total_prox_loss / num_batches,
    }

    return all_loss, model


def local_model_train_only_with_dp(local_trainset, model, config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    all_latent_vectors = []
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = config["local_epochs"]
    lambda_kl = config["lambda_kl"]
    lambda_lip = config["lambda_lip"]
    latent_dim = config["latent_dim"]
    folder = config["folder"]
    cid = config["cid"]
    server_round = config["server_round"]
    noise_multiplier = config["noise_multiplier"]
    local_dataloader = DataLoader(
        local_trainset, batch_size=config["batch_size"], shuffle=True
    )

    for epoch in range(num_epochs):
        total_train_loss = total_lip_loss = 0
        epoch_lip_constants = []
        batch_grad_norms = []  # Store gradient norms for calculating median

        for batch_idx, data in enumerate(local_dataloader):
            features = data[0].to(device)
            optimizer.zero_grad()

            recon_batch, mu, log_var = model(features)
            z = model.reparameterize(mu, log_var)
            all_latent_vectors.append(z.detach().cpu().numpy())
            vae_loss, _, _, _ = mse_loss_function_with_gp(
                model,
                features,
                recon_batch,
                mu,
                log_var,
                lambda_kl,
                use_gp=False,
                gradient_penalty_weight=lambda_lip,
            )

            vae_loss.backward()
            dp_params = [p for p in model.parameters() if hasattr(p, "grad_sample")]

            # Calculate gradient norms without clipping
            for param in dp_params:
                if hasattr(param, "grad_sample"):
                    grad_sample_norms = torch.norm(
                        param.grad_sample.view(param.grad_sample.shape[0], -1), dim=1
                    )
                    batch_grad_norms.extend(grad_sample_norms.cpu().tolist())

            # Calculate median norm for this batch
            if batch_grad_norms:
                max_grad_norm = float(np.median(batch_grad_norms))
                print(f"Median grad norm: {max_grad_norm}")
            else:
                max_grad_norm = config.get("max_grad_norm", 1.0)  # fallback value

            # clip gradients using median norm
            grad_norms = clip_grad_sample(dp_params, max_grad_norm)

            # Add noise scaled to median norm
            for param in dp_params:
                if hasattr(param, "grad_sample"):
                    noise = (
                        torch.randn_like(param.grad) * max_grad_norm * noise_multiplier
                    )
                    param.grad = torch.sum(param.grad_sample, dim=0) + noise
                    param.grad_sample = None

            optimizer.step()
            total_train_loss += vae_loss.item()

    # Fit GMM on collected latent vectors
    all_latent_vectors = np.concatenate(all_latent_vectors, axis=0)
    gmm = GaussianMixture(n_components=10, covariance_type="diag", random_state=42)
    gmm.fit(all_latent_vectors)
    save_gmm(gmm, f"{folder}/gmm_c_{cid}_r{server_round}.joblib")
    all_loss = {
        "local_total_train_loss": total_train_loss / len(local_dataloader),
    }
    return all_loss, model


def visualize_gen_image(
    net,
    testloader,
    device,
    rnd=None,
    folder=None,
    batch_size=None,
):
    """Validate the network on the entire test set."""
    os.makedirs(folder, exist_ok=True)
    net.eval()
    with torch.no_grad():
        for idx, data in enumerate(testloader):
            if type(data) == list:
                features = data[0]
                labels = data[1]
            images = features.to(device)
            client_ids = labels.to(device)
            break
        print(f" true {images.size()}")
        save_image(
            images.view(batch_size, 1, 28, 28), f"{folder}/true_img_at_{rnd}.png"
        )

        net.eval()

        generated_img, real_mu, real_logvar = net(images)

        print(f"gene {generated_img.size()}")
        save_image(
            generated_img.view(batch_size, 1, 28, 28),
            f"{folder}/test_generated_at_{rnd}.png",
        )
    return f"{folder}/true_img_at_{rnd}.png", f"{folder}/test_generated_at_{rnd}.png"


def visualize_latent_space(model, test_loader, device, filename=None, folder=None):
    # Set the backend to non-interactive Agg to avoid GUI warnings
    import matplotlib

    matplotlib.use("Agg")

    model.eval()
    z_points = []
    labels = []

    with torch.no_grad():
        for data, label in test_loader:
            data = data.to(device)
            _, mu, _ = model(data)
            z_points.append(mu.cpu().numpy())
            labels.append(label.numpy())

    z_points = np.concatenate(z_points, axis=0)
    labels = np.concatenate(labels, axis=0)

    # If latent dimension > 2, use PCA to reduce to 2D
    if z_points.shape[1] > 2:
        np.random.seed(42)  # For reproducibility
        pca = PCA(n_components=2)
        z_points = pca.fit_transform(z_points)
        print(f"Explained variance ratio: {pca.explained_variance_ratio_}")

    # Set figure and font sizes
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
        }
    )

    # Create figure with reasonable size
    plt.figure(figsize=(10, 8))

    # Create scatter plot
    scatter = plt.scatter(
        z_points[:, 0],
        z_points[:, 1],
        c=labels,
        cmap="tab10",
        s=50,  # Reduced marker size
        alpha=0.8,
    )

    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.ax.tick_params(labelsize=10)

    # Set labels and title
    plt.xlabel("PCA1" if z_points.shape[1] > 2 else "z[0]")
    plt.ylabel("PCA2" if z_points.shape[1] > 2 else "z[1]")
    plt.title(
        (
            "2D Latent Space (PCA Projection)"
            if z_points.shape[1] > 2
            else "2D Latent Space"
        ),
        pad=15,
    )

    # Enhance grid and ticks
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tick_params(axis="both", which="major", width=1.5, length=5)

    # Add tight layout before saving
    plt.tight_layout()

    # Save with adjusted quality settings
    plt.savefig(
        f"{folder}/{filename}.png",
        format="png",
        dpi=300,  # Reduced DPI
        bbox_inches="tight",
        pad_inches=0.1,
        facecolor="white",
    )
    plt.close()
    return f"{folder}/{filename}.png"


def sample_gmm_latent(gmm, n_samples, device, dtype=torch.float32):
    """
    Sample from GMM latent space
    """
    # Sample component indices based on mixture weights
    component_indices = np.random.choice(
        gmm.n_components, size=n_samples, p=gmm.weights_
    )

    # Get corresponding means and variances
    means = gmm.means_[component_indices].astype(
        np.float32
    )  # Shape: (n_samples, latent_dim)
    vars = gmm.covariances_[component_indices].astype(
        np.float32
    )  # Shape: (n_samples, latent_dim) for diagonal

    # Sample from normal distribution using selected means and variances
    z = torch.randn(n_samples, means.shape[1], device=device, dtype=dtype)
    z = z * torch.tensor(np.sqrt(vars), device=device, dtype=dtype) + torch.tensor(
        means, device=device, dtype=dtype
    )

    return z


def sample_from_model(model, device, num_samples, model_idx, server_round, folder):
    model.eval()  # Set the model to evaluation mode
    # set_seed(seed)
    gmm = load_gmm(f"{folder}/gmm_c_{model_idx}_r{server_round}.joblib")
    with torch.no_grad():
        if isinstance(model, (VAE, SimpleVAE)):
            z = sample_gmm_latent(gmm, num_samples, device)
            # z = torch.randn(num_samples, model.latent_dim).to(device)
            features = model.decode(z)
        else:  # GAN
            z = torch.randn(num_samples, model.generator.model[0].in_features).to(
                device
            )
            features = model.generator(z)
        # Apply thresholding
        # features = torch.where(features >= 0.8, 1.0, features)  # Values >= 0.8 become 1
        # features = torch.where(features <= 0.2, 0.0, features)  # Values <= 0.2 become 0

        labels = torch.full((num_samples,), model_idx, dtype=torch.long, device=device)
    return features, labels


def forward_hook(module, intsr, outtsr):
    module.input = intsr[0]


def backward_hook(module, grad_input, grad_output):
    grad_output = grad_output[0]
    grad_input = module.input

    A = grad_input
    B = grad_output
    if A.dtype != B.dtype:
        A = A.to(B.dtype)

    # Compute per-example gradients
    gs = torch.einsum("n...i,n...j->nij", B, A)
    if hasattr(module, "bias") and module.bias is not None:
        gs_bias = torch.einsum("n...k->nk", B)

    module.weight.grad_sample = gs.float()
    if hasattr(module, "bias") and module.bias is not None:
        module.bias.grad_sample = gs_bias.float()


def make_module_dp(module):
    """
    Adds hooks to compute per-sample gradients for the module
    """
    if isinstance(module, nn.Sequential):
        for layer in module:
            if isinstance(layer, nn.Linear):
                layer.register_forward_hook(forward_hook)
                layer.register_full_backward_hook(backward_hook)
    elif isinstance(module, nn.Linear):
        module.register_forward_hook(forward_hook)
        module.register_full_backward_hook(backward_hook)


def get_noise_multiplier(eps, delta, steps, sampling_prob, init_sigma=0.5):
    """Get noise multiplier using PRVAccountant"""
    current_sigma = init_sigma

    # Binary search with different precisions
    for precision in [1.0, 0.1, 0.01]:
        while current_sigma > 0:
            current_eps = get_epsilon_prv(current_sigma, delta, steps, sampling_prob)
            print(f"Current sigma: {current_sigma}, Current epsilon: {current_eps}")
            if current_eps < eps:
                current_sigma -= precision
            else:
                current_sigma += precision
                break

    if current_sigma <= 0.01:
        raise ValueError("Cannot find valid sigma for given privacy parameters")

    return current_sigma


def get_epsilon_prv(noise_multiplier, delta, steps, sampling_prob):
    """Calculate epsilon using PRVAccountant"""
    prv = PoissonSubsampledGaussianMechanism(
        noise_multiplier=noise_multiplier, sampling_probability=sampling_prob
    )
    accountant = PRVAccountant(
        prvs=[prv], max_self_compositions=[steps], eps_error=0.1, delta_error=delta / 10
    )
    _, _, eps_upper = accountant.compute_epsilon(
        delta=delta, num_self_compositions=[steps]
    )
    return eps_upper


def clip_grad_sample(params, max_norm):
    """Clip per-example gradients"""
    # Get batch size from first parameter
    for p in params:
        if hasattr(p, "grad_sample"):
            n = p.grad_sample.shape[0]
            break

    grad_norm_list = torch.zeros(n).cuda()

    # Calculate gradient norms
    for p in params:
        if hasattr(p, "grad_sample"):
            flat_g = p.grad_sample.reshape(n, -1)
            current_norm_list = torch.norm(flat_g, dim=1)
            grad_norm_list += torch.square(current_norm_list)

    grad_norm_list = torch.sqrt(grad_norm_list)

    # Calculate scaling factors
    scaling = max_norm / grad_norm_list
    scaling[scaling > 1] = 1

    # Apply scaling
    for p in params:
        if hasattr(p, "grad_sample"):
            p_dim = len(p.shape)

            scaling_view = scaling.view([n] + [1] * p_dim)
            p.grad_sample *= scaling_view
    return grad_norm_list


def save_gmm(gmm, filepath):
    """
    Save trained GMM model to disk

    Args:
        gmm: Trained GaussianMixture model
        filepath: Path to save the model (e.g., 'models/gmm.joblib')
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(gmm, filepath)
        print(f"GMM model saved successfully to {filepath}")
    except Exception as e:
        print(f"Error saving GMM model: {str(e)}")


def load_gmm(filepath):
    """
    Load trained GMM model from disk

    Args:
        filepath: Path to the saved model

    Returns:
        Loaded GaussianMixture model or None if loading fails
    """
    try:
        gmm = joblib.load(filepath)
        print(f"GMM model loaded successfully from {filepath}")
        return gmm
    except Exception as e:
        print(f"Error loading GMM model: {str(e)}")
        return None


def compute_FID_and_IS(
    testloader, vae_model, device, num_samples=None, max_batch_size=32
):
    """
    Memory-efficient implementation to calculate both FID score and Inception Score for VAE-generated images.

    Args:
        testloader: DataLoader containing test images
        vae_model: Trained VAE model
        device: torch device (cuda/cpu)
        num_samples: Optional; number of samples to generate. If None, uses same size as testloader
        max_batch_size: Maximum batch size to use for feature extraction to manage memory

    Returns:
        tuple: (fid_score, inception_score, inception_score_std)
    """

    import gc

    def clear_gpu_memory():
        """Helper function to clear GPU memory"""
        if device.type == "cuda":
            torch.cuda.empty_cache()
            gc.collect()

    # Set up inception model for FID
    inception = inception_v3(pretrained=True, transform_input=False)
    inception.fc = nn.Identity()
    inception.eval()
    inception.to(device)

    # Set up InceptionScore calculator
    inception_score_calc = InceptionScore(normalize=True).to(device)

    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.expand(3, -1, -1)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    def calculate_fid(real_features, generated_features):
        mu_r, sigma_r = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
        mu_g, sigma_g = generated_features.mean(axis=0), np.cov(
            generated_features, rowvar=False
        )

        diff = mu_r - mu_g
        covmean = sqrtm(sigma_r @ sigma_g)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        fid = np.sum(diff**2) + np.trace(sigma_r + sigma_g - 2 * covmean)
        return fid

    def process_batch(images, process_inception_score=True):
        """Process a batch of images and return features and processed images"""
        if isinstance(images, list):
            images = images[0]

        batch_processed = []
        for img in images:
            processed_img = transform(img)
            batch_processed.append(processed_img)

        batch_tensor = torch.stack(batch_processed).to(device)

        with torch.no_grad():
            features = inception(batch_tensor)

        if process_inception_score:
            inception_score_calc.update(batch_tensor)

        features_cpu = features.cpu().numpy()
        del features
        del batch_tensor
        clear_gpu_memory()

        return features_cpu

    def extract_features(dataloader, process_inception_score=True):
        all_features = []

        # Process in smaller batches
        for images in dataloader:
            if len(images) > max_batch_size:
                # Split into smaller batches
                for i in range(0, len(images), max_batch_size):
                    batch = images[i : i + max_batch_size]
                    features_cpu = process_batch(batch, process_inception_score)
                    all_features.append(features_cpu)
            else:
                features_cpu = process_batch(images, process_inception_score)
                all_features.append(features_cpu)

        return np.concatenate(all_features, axis=0)

    # Set models to eval mode
    vae_model.eval()

    # Extract real features from testloader for FID
    print("Processing real images...")
    real_features = extract_features(testloader)
    clear_gpu_memory()

    # If num_samples not specified, match the size of test set
    if num_samples is None:
        num_samples = len(testloader.dataset)

    # Generate images using the VAE in smaller batches
    print("Generating images...")
    generated_images = []
    batch_size = min(testloader.batch_size, max_batch_size)
    num_batches = (num_samples + batch_size - 1) // batch_size

    with torch.no_grad():
        for i in range(num_batches):
            current_batch_size = min(batch_size, num_samples - i * batch_size)
            z = torch.randn(current_batch_size, vae_model.latent_dim).to(device)
            generated = vae_model.decode(z).reshape(current_batch_size, 1, 28, 28)
            generated_images.append(generated.cpu())
            clear_gpu_memory()

    # Concatenate all generated images
    generated_images = torch.cat(generated_images, dim=0)

    # Create a DataLoader for the generated images with controlled batch size
    generated_loader = torch.utils.data.DataLoader(
        generated_images, batch_size=batch_size, shuffle=False
    )

    # Extract features from generated images for FID
    print("Processing generated images...")
    generated_features = extract_features(generated_loader)

    # Calculate FID
    print("Calculating FID score...")
    fid_score = calculate_fid(real_features, generated_features)

    # Calculate Inception Score
    print("Calculating Inception Score...")
    is_mean, is_std = inception_score_calc.compute()

    # Final cleanup
    clear_gpu_memory()

    return fid_score, is_mean.item(), is_std.item()


def compute_real_IS(testloader, device, max_batch_size=32):
    """
    Memory-efficient implementation to calculate Inception Score for real images.

    Args:
        testloader: DataLoader containing test images
        device: torch device (cuda/cpu)
        max_batch_size: Maximum batch size to use for feature extraction to manage memory

    Returns:
        tuple: (inception_score_mean, inception_score_std)
    """
    import gc
    import torch
    from torchvision.models import inception_v3
    from torchvision import transforms
    from torch import nn

    def clear_gpu_memory():
        """Helper function to clear GPU memory"""
        if device.type == "cuda":
            torch.cuda.empty_cache()
            gc.collect()

    # Set up InceptionScore calculator
    inception_score_calc = InceptionScore(normalize=True).to(device)

    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Lambda(
                lambda x: x.expand(3, -1, -1)
            ),  # Expand single channel to 3 channels
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    def process_batch(images):
        """Process a batch of images for Inception Score calculation"""
        if isinstance(images, list):
            images = images[0]

        batch_processed = []
        for img in images:
            processed_img = transform(img)
            batch_processed.append(processed_img)

        batch_tensor = torch.stack(batch_processed).to(device)

        with torch.no_grad():
            inception_score_calc.update(batch_tensor)

        del batch_tensor
        clear_gpu_memory()

    print("Processing images for Inception Score calculation...")

    # Process images in smaller batches
    for images in testloader:
        if len(images) > max_batch_size:
            # Split into smaller batches
            for i in range(0, len(images), max_batch_size):
                batch = images[i : i + max_batch_size]
                process_batch(batch)
        else:
            process_batch(images)

    # Calculate Inception Score
    print("Calculating Inception Score...")
    is_mean, is_std = inception_score_calc.compute()

    # Final cleanup
    clear_gpu_memory()

    return is_mean.item(), is_std.item()


def evaluate_vae_encoder_split(vae_model, test_loader, device):
    """
    Evaluate VAE encoder features using 80-20 split for training and evaluation.

    Args:
        vae_model: Trained VAE model
        test_loader: DataLoader for test data
        device: torch device (cuda/cpu)

    Returns:
        dict: Dictionary containing all metrics
    """
    vae_model.eval()
    all_features = []
    all_labels = []

    # Extract features using VAE encoder
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            # Get encoder outputs (mu, logvar)
            mu, _ = vae_model.encode(data.view(-1, 784))
            all_features.append(mu.cpu().numpy())
            all_labels.append(labels.numpy())

    # Convert to numpy arrays
    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Train classifier
    clf = LogisticRegression(max_iter=1000, multi_class="multinomial")
    clf.fit(X_train, y_train)

    # Make predictions
    y_pred = clf.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average="weighted"
    )

    # Get per-class metrics
    class_report = classification_report(y_test, y_pred, output_dict=True)

    # Compile metrics
    metrics = {
        "accuracy": accuracy,
        # "precision": precision,
        # "recall": recall,
        "f1": f1,
        "per_class_metrics": class_report,
    }

    # Print summary
    print("\nClassification Metrics (80% test split):")
    print(f"Accuracy: {accuracy*100:.2f}%")
    # print(f"Precision: {precision*100:.2f}%")
    # print(f"Recall: {recall*100:.2f}%")
    print(f"F1 Score: {f1*100:.2f}%")

    return metrics, clf


"""
# Load statistics
latent_stats = np.load(f"{folder}/latent_stats_c_{cid}_r{server_round}.npy", allow_pickle=True).item()

# Create visualization
import matplotlib.pyplot as plt
import seaborn as sns

def plot_latent_stats(stats, title):
    plt.figure(figsize=(15, 6))
    
    # Prepare data for box plot
    data = []
    labels = []
    for dim, dim_stats in enumerate(stats['dimensions']):
        data.append([dim_stats['min'], dim_stats['q1'], 
                    dim_stats['median'], dim_stats['q3'], 
                    dim_stats['max']])
        labels.append(f'Dim {dim}')
    
    # Create box plot
    plt.boxplot(data, labels=labels)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()

# Plot both mu and log_var statistics
plot_latent_stats(latent_stats['mu'], 'Distribution of μ across Latent Dimensions')
plot_latent_stats(latent_stats['log_var'], 'Distribution of log_var across Latent Dimensions')
"""
import copy

def standard_local_model_train_moon(
    local_trainset, model, config, global_model, prev_models
):
    """
    Train a local model using MOON (Model-Contrastive) approach with VAEs

    Args:
        local_trainset: Local training dataset
        model: Current local model (VAE)
        config: Training configuration
        global_model: Global model for contrastive learning
        prev_models: List of previous models for negative samples

    Returns:
        dict: Training losses
        model: Trained model
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    global_model = copy.deepcopy(model)
    global_model.to(device)
    model.train()
    global_model.eval()
    all_latent_vectors = []

    # Set up optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Get training parameters from config
    num_epochs = config["local_epochs"]
    lambda_kl = config["lambda_kl"]
    lambda_lip = config["lambda_lip"]
    mu = config.get("mu", 1.0)  # Coefficient for contrastive loss, default 1.0
    temperature = config.get(
        "temperature", 0.5
    )  # Temperature for contrastive loss, default 0.5
    latent_dim = config["latent_dim"]
    cid = config.get("cid", 0)
    folder = config.get("folder", "")
    server_round = config.get("server_round", 0)

    # Set up dataloader
    local_dataloader = DataLoader(
        local_trainset, batch_size=config["batch_size"], shuffle=True
    )

    # Move previous models to device
    prev_models_device = []
    for prev_model in prev_models:
        prev_models_device.append(prev_model.to(device))

    # Track all losses
    total_train_loss = 0
    total_mse = 0
    total_kl_loss = 0
    total_lip_loss = 0
    total_contrastive_loss = 0
    num_batches = 0
    epoch_lip_constants = []

    # Define cosine similarity function for contrastive loss
    cos = torch.nn.CosineSimilarity(dim=-1)

    for epoch in range(num_epochs):
        epoch_train_loss = 0
        epoch_mse = 0
        epoch_kl_loss = 0
        epoch_lip_loss = 0
        epoch_contrastive_loss = 0
        batch_count = 0

        for batch_idx, data in enumerate(local_dataloader):
            features = data[0].to(device)

            optimizer.zero_grad()

            # Forward pass through current model
            recon_batch, mu, log_var = model(features)
            z = model.reparameterize(mu, log_var)
            all_latent_vectors.append(z.detach().cpu().numpy())

            # Forward pass through global model (for contrastive loss)
            with torch.no_grad():
                _, global_mu, global_log_var = global_model(features)
                global_z = global_model.reparameterize(global_mu, global_log_var)

            # Compute VAE loss
            vae_loss, mse, kl, lip_loss = mse_loss_function_with_gp(
                model,
                features,
                recon_batch,
                mu,
                log_var,
                lambda_kl,
                use_gp=True if lambda_lip > 0 else False,
                gradient_penalty_weight=lambda_lip,
            )

            # Get embeddings from previous models for negative samples
            prev_z_list = []
            for prev_model in prev_models_device:
                with torch.no_grad():
                    _, prev_mu, prev_log_var = prev_model(features)
                    prev_z = prev_model.reparameterize(prev_mu, prev_log_var)
                    prev_z_list.append(prev_z)

            # Compute contrastive loss
            contrastive_loss = torch.tensor(0.0, device=device)
            if len(prev_z_list) > 0:  # Only compute if we have previous models
                # Compute cosine similarity between current and global embeddings
                pos_sim = cos(z, global_z).unsqueeze(-1)  # Shape: [batch_size, 1]

                # Debug: check similarity values
                print(
                    f"Positive similarity stats: min={pos_sim.min().item():.4f}, max={pos_sim.max().item():.4f}, mean={pos_sim.mean().item():.4f}"
                )

                # Compute cosine similarities with previous models (negative samples)
                neg_sims = []
                for i, prev_z in enumerate(prev_z_list):
                    neg_sim = cos(z, prev_z).unsqueeze(-1)  # Shape: [batch_size, 1]
                    print(
                        f"Negative similarity {i} stats: min={neg_sim.min().item():.4f}, max={neg_sim.max().item():.4f}, mean={neg_sim.mean().item():.4f}"
                    )
                    neg_sims.append(neg_sim)

                # Concatenate all similarities
                if neg_sims:  # Check if we have any negative samples
                    neg_sims = torch.cat(
                        neg_sims, dim=1
                    )  # Shape: [batch_size, num_prev_models]
                    logits = torch.cat(
                        [pos_sim, neg_sims], dim=1
                    )  # Shape: [batch_size, 1+num_prev_models]

                    # Apply temperature scaling
                    logits /= temperature

                    # Labels: positive pair is the first one (index 0)
                    labels = torch.zeros(features.size(0)).to(logits.device).long()

                    # Compute cross entropy loss
                    contrastive_loss = F.cross_entropy(logits, labels)
            # Total loss: VAE loss + weighted contrastive loss
            total_loss = vae_loss + mu * contrastive_loss
            print(
                f"Epoch {epoch}, Batch {batch_idx}, VAE Loss: {vae_loss.item()}, Contrastive Loss: {contrastive_loss.item()}"
            )
            # Ensure total_loss is a scalar for backward pass
            if not torch.is_tensor(total_loss) or total_loss.dim() > 0:
                total_loss = total_loss.sum()

            # Backward pass
            total_loss.backward()
            optimizer.step()

            # Track losses
            epoch_train_loss += total_loss.item()
            epoch_mse += mse.item()
            epoch_kl_loss += kl.item()
            epoch_lip_loss += lip_loss.item() if lambda_lip > 0 else 0
            epoch_contrastive_loss += contrastive_loss.item()
            batch_count += 1
            num_batches += 1

        # Average losses for the epoch
        epoch_train_loss /= batch_count
        epoch_mse /= batch_count
        epoch_kl_loss /= batch_count
        epoch_lip_loss /= batch_count if batch_count > 0 else 1
        epoch_contrastive_loss /= batch_count if batch_count > 0 else 1

        # Add to total losses
        total_train_loss += epoch_train_loss
        total_mse += epoch_mse
        total_kl_loss += epoch_kl_loss
        total_lip_loss += epoch_lip_loss
        total_contrastive_loss += epoch_contrastive_loss

        # Compute and log Lipschitz constant if needed
        if lambda_lip > 0:
            lip_constant = compute_epoch_lipschitz_constant(model)
            epoch_lip_constants.append(lip_constant)
            print(f"Epoch {epoch} Lipschitz constant: {lip_constant}")

    # Calculate average losses
    num_epochs_float = float(num_epochs)
    avg_train_loss = total_train_loss / num_epochs_float
    avg_mse = total_mse / num_epochs_float
    avg_kl_loss = total_kl_loss / num_epochs_float
    avg_lip_loss = total_lip_loss / num_epochs_float
    avg_contrastive_loss = total_contrastive_loss / num_epochs_float

    # Fit GMM on collected latent vectors for sampling (if we have any)
    if all_latent_vectors:
        all_latent_vectors = np.concatenate(all_latent_vectors, axis=0)
        from sklearn.mixture import GaussianMixture
        import joblib

        gmm = GaussianMixture(n_components=10, covariance_type="diag", random_state=42)
        gmm.fit(all_latent_vectors)

        # Save GMM model if folder is provided
        if folder:
            joblib.dump(gmm, f"{folder}/gmm_c_{cid}_r{server_round}.joblib")

    # Return all losses
    all_loss = {
        "local_total_train_loss": avg_train_loss,
        "local_total_mse": avg_mse,
        "local_total_kl_loss": avg_kl_loss,
        "local_total_gp_loss": avg_lip_loss,
        "local_total_contrastive_loss": avg_contrastive_loss,
    }

    # Move previous models back to CPU to save memory
    for prev_model in prev_models_device:
        prev_model.to("cpu")

    global_model.to("cpu")

    return all_loss, model
