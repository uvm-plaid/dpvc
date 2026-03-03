import dpvc
import torch
import numpy as np
from tqdm import tqdm

device="cuda:0" if torch.cuda.is_available() else "cpu"


def train_autoencoder(model, embeddings, epochs=1000, labels=None):
    BATCH_SIZE = min(256, len(embeddings))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    outputs = []
    losses = []
    beta = 1

    if labels is not None:
        num_labels = len(labels.keys())
        label_vals = [list(labels[f]) for f in labels]
        label_tensor = torch.tensor(label_vals).to(embeddings.device).squeeze().T
    else:
        label_tensor = torch.tensor(0 for _ in embeddings)

    print(f'Training controllable autoencoder for {epochs} epochs...')
    for epoch in tqdm(range(epochs)):
        with torch.no_grad():
            indexes = torch.randperm(embeddings.shape[0])
            embeddings_batches = torch.split(embeddings[indexes], BATCH_SIZE)
            labels_batches = torch.split(label_tensor[indexes], BATCH_SIZE)

        for embeddings_b, labels_b in zip(embeddings_batches, labels_batches):
            optimizer.zero_grad()

            reconstructed = model(embeddings_b)
            recon_loss = ((embeddings_b - reconstructed)**2).sum()
            if labels is not None:
                label_loss = beta*((model.last_z[:, :num_labels] - labels_b)**2).sum()
            else:
                label_loss = 0
            kl_loss = beta*model.kl
            loss = recon_loss + kl_loss + label_loss

            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            print('loss:', loss.item(), recon_loss.item(), kl_loss.item(), label_loss.item())

    print('Ending loss:', loss.item())


data = torch.load('embeddings/openvoice_embeddings_features.pt')
embeddings = data['data'].to(device).squeeze()
ages = data['ages']
genders = data['genders']
print('Shape of extracted embeddings:', embeddings.shape)

# Construct the VC system wrapper
vc_wrapper = dpvc.OpenVoiceWrapper()

# Train the VAE
AE = dpvc.VariationalAutoencoder(latent_dims=8).to(device)
train_autoencoder(AE, embeddings, epochs=200, labels={'ages': ages, 'genders': genders})
torch.save(AE.state_dict(), 'openvoice_vae_features.pt')
