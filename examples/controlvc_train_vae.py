import dpvc
import torch

device="cuda:0" if torch.cuda.is_available() else "cpu"


embeddings = torch.load('embeddings/controlvc_emb.pt')['data']
print('Shape of extracted embeddings:', embeddings.shape)

# Train the VAE
AE = dpvc.VariationalAutoencoder(latent_dims=6).to(device)
dpvc.utils.train_autoencoder(AE, embeddings, epochs=1000)
torch.save(AE.state_dict(), 'controlvc_vae.pt')
