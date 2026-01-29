import dpvc

#src_path = 'trump_0.wav'
src_path = 'joe.wav'
ae_path = 'naturalspeech3_vae.pt'
#ae_path = None

vc_wrapper = dpvc.NaturalSpeech3Wrapper()
anonymizer = dpvc.Anonymizer(vc_wrapper, vae_checkpoint_path=ae_path)

emb_src = vc_wrapper.extract_embedding('joe.wav')
emb_trg = vc_wrapper.extract_embedding('trump_0.wav')
vc_wrapper.inference('joe.wav', 'new_joe.wav', emb_src, emb_trg)
