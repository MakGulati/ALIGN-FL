import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
from torchvision.utils import save_image
from ..utils.gen_utils import VAE, loss_function, mse_loss_function_with_gp
import random
import numpy as np


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def train_global_vae(trainset, model, config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.train()
    vae_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # scheduler = ReduceLROnPlateau(vae_optimizer, "min", patience=5, factor=0.5)

    num_epochs = config["global_epochs"]
    lambda_kl = config["global_lambda_kl"]
    global_dataloader = DataLoader(
        trainset, batch_size=config["global_batch_size"], shuffle=True
    )

    for epoch in range(num_epochs):
        train_loss = 0
        for batch_idx, data in enumerate(global_dataloader):
            features = data[0].to(device)
            tasks = data[1].to(device)

            vae_optimizer.zero_grad()
            recon_batch, mu, logvar = model(features)

            # vae_loss = loss_function(
            #     recon_batch, features, mu, logvar, lambda_kl=lambda_kl, normalize=False
            # )
            vae_loss, _, _, _ = mse_loss_function_with_gp(
                model, features, recon_batch, mu, logvar, BETA=lambda_kl, use_gp=False,
            )

            vae_loss.backward()
            vae_optimizer.step()

            train_loss += vae_loss.item()
        # scheduler.step(train_loss / len(global_dataloader))
        # print(
        #     f"loss at server on ep: {epoch}: {train_loss / len(global_dataloader.dataset)}"
        # )
        print(f"loss at server on ep: {epoch}: {train_loss / len(global_dataloader)}")

    return model


def sample_and_visualize(
    net,
    device,
    num_samples,
    folder,
    rnd=None,
    latent_dim=None,
    image_size=(28, 28),
):

    # Set the fixed seed for reproducibility
    # set_seed(seed)
    net.eval()

    # Sample from standard Gaussian
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim).to(device)
        samples = net.decoder(z)

    # Reshape samples to image format
    images = samples.view(num_samples, 1, image_size[0], image_size[1])

    # Save the images
    filename = f"sampled_img_at_{rnd}.png"
    save_path = os.path.join(folder, filename)
    save_image(images, save_path)

    return save_path



