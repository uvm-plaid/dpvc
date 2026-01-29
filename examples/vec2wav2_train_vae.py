import dpvc
import torch

device="cuda:0" if torch.cuda.is_available() else "cpu"

embeddings = torch.load('embeddings/vec2wav2_emb.pt')['data'].to(device)
print('Shape of extracted embeddings:', embeddings.shape)

# Construct the VC system wrapper
vc_wrapper = dpvc.Vec2Wav2Wrapper()

# Train the VAE
AE = dpvc.VariationalAutoencoder(input_dim=1024, latent_dims=16).to(device)
dpvc.utils.train_autoencoder(AE, embeddings, epochs=2000)
torch.save(AE.state_dict(), 'vec2wav2_vae.pt')
