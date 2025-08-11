import torch
import torch.nn as nn
from model_embedding_vae import *
import pickle

BATCH_SIZE = 512

device='cuda'

emb = torch.load('all_emb_labeled_cv_full.pt')['data'].to(device).squeeze()
print(emb.shape)

num_emb, dim_emb = emb.shape

# Model Initialization
# model = VariationalAutoencoder(6)  #test 1
# model = VariationalAutoencoder(latent_dims=32) #test 32
model = VariationalAutoencoder(latent_dims=6) #test 32
model.to(device)
#model.set_noise_mult(16)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)#, weight_decay=1e-7)

epochs = 5000
#epochs = 300
outputs = []
losses = []
beta = 1


for epoch in range(epochs):
    with torch.no_grad():
        indexes = torch.randperm(emb.shape[0])
        emb = emb[indexes]
        emb_batches = torch.split(emb, BATCH_SIZE)

    for emb_b in emb_batches:

        optimizer.zero_grad()

        reconstructed = model(emb_b)
        recon_loss = ((emb_b - reconstructed)**2).sum()
        kl_loss = beta*model.encoder.kl
        loss = recon_loss + kl_loss

        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        print(f'epoch {epoch} loss: {loss.item()/BATCH_SIZE:.5f}, recon: {recon_loss.item()/BATCH_SIZE:.5f}, kl: {kl_loss.item()/BATCH_SIZE:.5f}')
        torch.save(model, 'embedding_vae.pt')


torch.save(model, 'embedding_vae.pt')
