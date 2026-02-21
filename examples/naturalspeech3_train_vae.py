import dpvc
import torch

device="cuda:0" if torch.cuda.is_available() else "cpu"

# Extract embeddings
embeddings = torch.load('naturalspeech_emb.pt')['data'].to(device)
print('Shape of extracted embeddings:', embeddings.shape)

# Train the VAE
AE = dpvc.VariationalAutoencoder(latent_dims=6).to(device)
dpvc.utils.train_autoencoder(AE, embeddings, epochs=2000)
torch.save(AE.state_dict(), 'naturalspeech3_vae.pt')
