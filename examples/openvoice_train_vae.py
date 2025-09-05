import dpvc
import torch

device="cuda:0" if torch.cuda.is_available() else "cpu"
files = ['output/openvoice_noisy_0.wav',
         'output/openvoice_noisy_1.wav',
         'output/openvoice_noisy_2.wav',
         'output/openvoice_noisy_3.wav',
         'output/openvoice_noisy_4.wav',
         'output/openvoice_noisy_5.wav',
         'output/openvoice_noisy_6.wav',
         'output/openvoice_noisy_7.wav',
         'output/openvoice_noisy_8.wav',
         'output/openvoice_noisy_9.wav',
         ]

# Construct the VC system wrapper
vc_wrapper = dpvc.OpenVoiceDPWrapper()

# Extract embeddings
embeddings = dpvc.utils.extract_embeddings(vc_wrapper, files).to(device)
print('Shape of extracted embeddings:', embeddings.shape)

# Train the VAE
AE = dpvc.VariationalAutoencoder(latent_dims=6).to(device)
dpvc.utils.train_autoencoder(AE, embeddings)
torch.save(AE.state_dict(), 'example_openvoice_vae.pt')
