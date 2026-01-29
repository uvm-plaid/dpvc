import dpvc
import torch

device="cuda:0" if torch.cuda.is_available() else "cpu"

data = torch.load('embeddings/openvoice_embeddings_features.pt')
embeddings = data['data'].to(device).squeeze()
ages = data['ages']
genders = data['genders']
print('Shape of extracted embeddings:', embeddings.shape)

# Construct the VC system wrapper
vc_wrapper = dpvc.OpenVoiceWrapper()

# Train the VAE
AE = dpvc.VariationalAutoencoder(latent_dims=16).to(device)
dpvc.utils.train_autoencoder(AE, embeddings, epochs=2000, labels={'ages': ages, 'genders': genders})
torch.save(AE.state_dict(), 'openvoice_vae_features.pt')
