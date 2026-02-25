import dpvc
import torch

device="cuda:0" if torch.cuda.is_available() else "cpu"

embeddings = torch.load('embeddings/openvoice_random_embeddings_cv.pt').to(device)
print('Shape of extracted embeddings:', embeddings.shape)

# Construct the VC system wrapper
vc_wrapper = dpvc.OpenVoiceWrapper()

# Train the VAE
AE = dpvc.VariationalAutoencoder(latent_dims=6).to(device)
dpvc.utils.train_autoencoder(AE, embeddings, epochs=500)
torch.save(AE.state_dict(), 'openvoice_vae.pt')
