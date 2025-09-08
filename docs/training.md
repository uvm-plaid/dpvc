# Training the Variational Autoencoder

DPVC works by perturbing a low-dimensional latent representation of
the speaker embedding. We learn this representation by training a
variational autoencoder (VAE) using speaker embeddings extracted from
training data using a voice control system.

## Extracting Embeddings

To extract the embeddings for training the VAE, construct a wrapper
around a voice control system and call the `extract_embeddings`
function. This function calls `extract_embedding` method of the
wrapper for each of the input files given in the second argument. It
returns a PyTorch tensor containing all of the extracted embeddings.

``` py
import dpvc
files = [...]

# Construct the VC system wrapper
vc_wrapper = dpvc.OpenVoiceWrapper()

# Extract embeddings
embeddings = dpvc.utils.extract_embeddings(vc_wrapper, files)
```

## Training the Autoencoder

The VAE itself is a standard PyTorch model, provided in the
`VariationalAutoencoder` class. To train the VAE, instantiate the
model object and then train it using a standard PyTorch optimizer. The
`train_autoencoder` function provides a standard training loop for
this purpose. After training, the model weights can be saved using
`torch.save`.

``` py
# Instantiate the VAE model
AE = dpvc.VariationalAutoencoder(latent_dims=6)

# Train the VAE
dpvc.utils.train_autoencoder(AE, embeddings, epochs=10000)

# Save the VAE weights
torch.save(AE.state_dict(), 'example_openvoice_vae.pt')
```

## Using the new Autoencoder

To use the trained VAE, pass the filename used when saving the model
weights in the optional `vae_checkpoint_path` argument when
constructing the anonymizer:

``` py
# Construct the anonymizer, using the newly trained VAE
anonymizer = dpvc.Anonymizer(vc_wrapper, vae_checkpoint_path='example_openvoice_vae.pt')
```
