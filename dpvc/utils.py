import torch
import random
import numpy as np
from tqdm import tqdm

def set_seed(seed):
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)


def train_autoencoder(model, embeddings, epochs=1000):
    BATCH_SIZE = min(64, len(embeddings))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)#, weight_decay=1e-7)
    outputs = []
    losses = []
    beta = 1

    print(f'Training autoencoder for {epochs} epochs...')
    for epoch in tqdm(range(epochs)):
        with torch.no_grad():
            indexes = torch.randperm(embeddings.shape[0])
            embeddings = embeddings[indexes]
            embeddings_batches = torch.split(embeddings, BATCH_SIZE)

        for embeddings_b in embeddings_batches:
            optimizer.zero_grad()

            reconstructed = model(embeddings_b)
            recon_loss = ((embeddings_b - reconstructed)**2).sum()
            kl_loss = beta*model.encoder.kl
            loss = recon_loss + kl_loss

            loss.backward()
            optimizer.step()

    print('Ending loss:', loss.item())
