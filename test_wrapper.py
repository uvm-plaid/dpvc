from pathlib import Path
from dpvc import ControlVCWrapper

print("="*50)
print("Testing ControlVC Wrapper")
print("="*50)

try:
    # Initialize
    print("\n1. Initializing wrapper...")
    wrapper = ControlVCWrapper(
        repo_root=Path("/Users/steve/repos/control-vc"),
        device="cpu",
        verbose=True
    )
    print("✓ Initialization successful!")

    # Test with example audio (if it exists)
    test_audio = Path("examples/trump_0.wav")
    if test_audio.exists():
        print(f"\n2. Testing embedding extraction with {test_audio}...")
        embedding = wrapper.extract_embedding(test_audio)
        print(f"✓ Embedding extracted: shape={embedding.shape}, dtype={embedding.dtype}")
        print(f"  Range: [{embedding.min().item():.3f}, {embedding.max().item():.3f}]")

        print(f"\n3. Testing voice conversion...")
        output = wrapper.infer(test_audio, embedding, out_sr=16000)
        print(f"✓ Conversion successful: shape={output.shape}, dtype={output.dtype}")

        # Save using soundfile instead of torchaudio to avoid torchcodec dependency
        import soundfile as sf
        audio_np = output.cpu().squeeze().numpy()
        sf.write("test_output.wav", audio_np, 16000)
        print(f"✓ Saved output to test_output.wav")
    else:
        print(f"\n⚠ Test audio not found at {test_audio}")
        print("  Skipping audio tests")

    print("\n" + "="*50)
    print("All tests passed! ✓")
    print("="*50)

except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
