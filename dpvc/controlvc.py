# dpvc/controlvc.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, Any, Dict, Tuple, Union
import sys
import json
import warnings
import numpy as np
import torch
import torchaudio
import librosa
import soundfile as sf
import os

class ControlVCWrapper:
    """
    Control-VC wrapper with a two-stage API for differential privacy:
      1) extract_embedding(wav_path) -> torch.Tensor (speaker embedding)
      2) inference(source_file, output_file, source_embedding, target_embedding)

    This wrapper directly loads ControlVC models and bypasses script-based execution
    for better integration with the DP anonymization pipeline.

    Expected checkpoints structure:
      checkpoints/
        ├── embed_f0stat2/          # Main VC model
        │   ├── config.json
        │   └── g_XXXXXXXX
        ├── 3000000-BL.ckpt         # Speaker embedding model
        ├── hubert_base_ls960.pt    # HuBERT model (optional for content extraction)
        └── km.bin                  # K-means quantizer (optional)
    """
    def __init__(
        self,
        device: str = "cuda",
        checkpoints_dir: Optional[Path] = None,
        config: Optional[Dict[str, Any]] = None,
        verbose: bool = False,
    ):
        repo_root = os.environ['CONTROLVC_PATH']
        self.repo_root = Path(repo_root).expanduser().resolve()
        self.verbose = verbose

        # Set device
        if device == "cpu" or not torch.cuda.is_available():
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.checkpoints_dir = (Path(checkpoints_dir).resolve()
                                if checkpoints_dir else (self.repo_root / "checkpoints"))
        self.config = config or {}

        if not self.repo_root.exists():
            raise FileNotFoundError(f"Control-VC repo root not found: {self.repo_root}")

        self._print(f"Initializing ControlVC wrapper with device: {self.device}")

        # Add repo to path for imports
        sys.path.insert(0, str(self.repo_root))

        # Load models
        self._load_models()

    def get_vae_config(self):
        local_path = os.path.dirname(os.path.abspath(__file__))
        vae_path = f'{local_path}/controlvc_embedding_vae.pt'

        config = {
            'checkpoint_path': vae_path,
            'latent_dim': 6,
            'input_dim': 256,
            'clip_threshold': 10.0,
            'post_clip_threshold': 10.0
        }
        return config

    def _print(self, msg: str):
        """Print if verbose mode enabled."""
        if self.verbose:
            print(f"[ControlVC] {msg}")

    def _load_models(self):
        """Load all required ControlVC models."""
        try:
            # Import ControlVC modules
            from models import CodeGenerator, D_VECTOR
            from dataset import get_yaapt_f0, mel_spectrogram, MAX_WAV_VALUE
            from utils import AttrDict, load_checkpoint, scan_checkpoint

            self._CodeGenerator = CodeGenerator
            self._D_VECTOR = D_VECTOR
            self._get_yaapt_f0 = get_yaapt_f0
            self._mel_spectrogram = mel_spectrogram
            self._MAX_WAV_VALUE = MAX_WAV_VALUE
            self._AttrDict = AttrDict
            self._load_checkpoint = load_checkpoint
            self._scan_checkpoint = scan_checkpoint

        except ImportError as e:
            raise ImportError(
                f"Failed to import ControlVC modules from {self.repo_root}. "
                f"Make sure the repo contains models.py, dataset.py, and utils.py. Error: {e}"
            )

        # Load main VC generator
        self._load_generator()

        # Load speaker embedding model
        self._load_speaker_model()

        self._print("All models loaded successfully")

    def _load_generator(self):
        """Load the main voice conversion CodeGenerator model."""
        # Find main VC checkpoint directory
        main_ckpt_dir = self.checkpoints_dir / "embed_f0stat2"
        if not main_ckpt_dir.exists():
            # Try to find any directory with config.json
            candidates = list(self.checkpoints_dir.glob("*/config.json"))
            if candidates:
                main_ckpt_dir = candidates[0].parent
                self._print(f"Using checkpoint dir: {main_ckpt_dir.name}")
            else:
                raise FileNotFoundError(
                    f"No ControlVC model checkpoint found in {self.checkpoints_dir}. "
                    "Expected directory with config.json (e.g., embed_f0stat2/)"
                )

        # Load config
        config_file = main_ckpt_dir / "config.json"
        with open(config_file) as f:
            json_config = json.loads(f.read())
        self.h = self._AttrDict(json_config)

        # Fix relative paths in config to be absolute based on checkpoints_dir
        if self.h.get('f0_quantizer_path'):
            f0_path = self.h.f0_quantizer_path
            # Remove leading "checkpoints/" if it exists since we'll add it via checkpoints_dir
            if f0_path.startswith('checkpoints/'):
                f0_path = f0_path[len('checkpoints/'):]

            f0_path = Path(f0_path)
            if not f0_path.is_absolute():
                # Convert relative path to absolute based on checkpoints_dir
                absolute_path = self.checkpoints_dir / f0_path
                if not absolute_path.exists():
                    # If it doesn't exist, disable F0 quantizer
                    warnings.warn(
                        f"F0 quantizer checkpoint not found at {absolute_path}. "
                        "Disabling F0 quantizer. This may affect voice quality."
                    )
                    self.h.f0_quantizer_path = None
                else:
                    self.h.f0_quantizer_path = str(absolute_path)

        # Create generator
        self.generator = self._CodeGenerator(self.h).to(self.device)

        # Load checkpoint - try multiple patterns
        cp_g = self._scan_checkpoint(str(main_ckpt_dir), 'g_')
        if cp_g is None:
            # Try with common extensions
            import glob
            patterns = [
                str(main_ckpt_dir / 'g_*.pth'),
                str(main_ckpt_dir / 'g_*.pt'),
                str(main_ckpt_dir / 'g_*.zip'),
            ]
            for pattern in patterns:
                matches = glob.glob(pattern)
                if matches:
                    cp_g = sorted(matches)[-1]
                    break

        if cp_g is None:
            raise FileNotFoundError(
                f"No generator checkpoint found in {main_ckpt_dir}. "
                f"Looked for g_*.pth, g_*.pt, g_*.zip, or g_????????"
            )

        state_dict_g = self._load_checkpoint(cp_g, device=str(self.device))
        self.generator.load_state_dict(state_dict_g['generator'])
        self.generator.eval()
        self.generator.remove_weight_norm()

        self._print(f"Loaded generator: {Path(cp_g).name}")

    def _load_speaker_model(self):
        """Load the D_VECTOR speaker embedding model."""
        # Find speaker model checkpoint
        spk_ckpt = self.checkpoints_dir / "3000000-BL.ckpt"
        if not spk_ckpt.exists():
            # Try to find any .ckpt file
            ckpts = list(self.checkpoints_dir.glob("*.ckpt"))
            if ckpts:
                spk_ckpt = ckpts[0]
                self._print(f"Using speaker checkpoint: {spk_ckpt.name}")
            else:
                warnings.warn(
                    f"Speaker embedding model not found in {self.checkpoints_dir}. "
                    "extract_embedding() will not work."
                )
                self.speaker_model = None
                return

        # Create and load model
        self.speaker_model = self._D_VECTOR(
            num_layers=3,
            dim_input=80,
            dim_cell=768,
            dim_emb=256
        ).to(self.device)

        checkpoint = torch.load(spk_ckpt, map_location=self.device)

        # Handle different checkpoint formats
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'model_b' in checkpoint:
            # AutoVC format - strip 'module.' prefix from keys
            from collections import OrderedDict
            state_dict = OrderedDict()
            for key, val in checkpoint['model_b'].items():
                new_key = key[7:] if key.startswith('module.') else key
                state_dict[new_key] = val
        else:
            raise KeyError(
                f"Unknown checkpoint format. Expected 'model' or 'model_b' key, "
                f"got: {list(checkpoint.keys())}"
            )

        self.speaker_model.load_state_dict(state_dict)
        self.speaker_model.eval()

        self._print(f"Loaded speaker model: {spk_ckpt.name}")

    # ---------- Public API ----------
    @torch.inference_mode()
    def extract_embedding(self, wav_path: Path, num_utterances: int = 1) -> torch.Tensor:
        """
        Extract speaker embedding from audio file using D_VECTOR model.

        Args:
            wav_path: Path to audio file
            num_utterances: Number of utterances to average (currently only supports 1)

        Returns:
            Speaker embedding tensor of shape (256,) or (256, 1) for ControlVC compatibility
        """
        if self.speaker_model is None:
            raise RuntimeError(
                "Speaker embedding model not loaded. "
                f"Ensure 3000000-BL.ckpt exists in {self.checkpoints_dir}"
            )

        wav_path = Path(wav_path).expanduser().resolve()
        if not wav_path.exists():
            raise FileNotFoundError(f"Reference wav not found: {wav_path}")

        self._print(f"Extracting embedding from {wav_path.name}")

        # Load and preprocess audio
        audio, sr = librosa.load(str(wav_path), sr=self.h.sampling_rate, mono=True)
        audio = librosa.util.normalize(audio) * 0.95

        # Compute mel spectrogram
        audio_tensor = torch.FloatTensor(audio).unsqueeze(0)
        mel = self._mel_spectrogram(
            audio_tensor,
            self.h.n_fft,
            self.h.num_mels,
            self.h.sampling_rate,
            self.h.hop_size,
            self.h.win_size,
            self.h.fmin,
            self.h.fmax
        )

        # Transpose to (batch, time, mel_bins) for D_VECTOR
        mel = mel.squeeze(0).transpose(0, 1).unsqueeze(0).to(self.device)

        # Extract embedding
        embedding = self.speaker_model(mel)  # Shape: (1, 256)

        # Return in format expected by ControlVC (can be (256,) or (256, 1))
        return embedding.squeeze(0).unsqueeze(-1).T  # Shape: (256, 1)

    @torch.inference_mode()
    def inference(self, source_file: Union[str, Path], output_file: Union[str, Path],
                  source_embedding: torch.Tensor, target_embedding: torch.Tensor) -> None:
        """
        Perform voice conversion using precomputed (possibly DP-noised) speaker embedding.

        Args:
            source_file: Path to source audio file.
            output_file: Path to write converted audio (16 kHz WAV).
            source_embedding: Speaker embedding of the source audio.
            target_embedding: Target speaker embedding. Accepts shapes
                ``(256,)``, ``(1, 256)``, or ``(256, 1)``.
        """
        out_sr = 16000
        pitch_shift = 1.0
        source_wav = Path(source_file).expanduser().resolve()
        if not source_wav.exists():
            raise FileNotFoundError(f"Source wav not found: {source_wav}")

        self._print(f"Converting {source_wav.name}")

        # Ensure target embedding is on correct device and shape.
        # Generator's _upsample expects (batch, dim) or (batch, dim, 1),
        # where batch=1 and dim=256.
        target_embedding = self._ensure_tensor(target_embedding)
        if target_embedding.dim() == 1:
            target_embedding = target_embedding.unsqueeze(0)       # (256,) -> (1, 256)
        elif target_embedding.dim() == 2 and target_embedding.shape[0] != 1:
            target_embedding = target_embedding.squeeze(-1).unsqueeze(0)  # (256, 1) -> (1, 256)
        # (1, 256) and (1, 256, 1) are already correct

        # Load and preprocess source audio
        audio, sr = librosa.load(str(source_wav), sr=self.h.sampling_rate, mono=True)
        audio = librosa.util.normalize(audio) * 0.95

        # Extract content codes (using pre-computed or on-the-fly HuBERT)
        codes = self._extract_content_codes(audio, sr)

        # Extract F0
        f0 = self._extract_f0(audio, sr)

        # Apply pitch shift if specified
        if pitch_shift != 1.0:
            f0[f0 != 0] *= pitch_shift

        # Prepare inputs for generator
        code_dict = {
            'code': torch.from_numpy(codes).long().unsqueeze(0).to(self.device),
            'f0': torch.from_numpy(f0).float().to(self.device),
            'spk_embed': target_embedding.to(self.device)  # (1, 256, 1)
        }

        # Add F0 stats if required by model config
        if self.h.get('f0_feats', False):
            # Calculate F0 mean and std from the current audio
            f0_flat = f0.flatten()
            f0_voiced = f0_flat[f0_flat != 0]
            if len(f0_voiced) > 0:
                f0_mean = float(f0_voiced.mean())
                f0_std = float(f0_voiced.std())
            else:
                # Fallback values if no voiced segments
                f0_mean = 0.0
                f0_std = 1.0

            code_dict['f0_stats'] = torch.FloatTensor([[f0_mean, f0_std]]).to(self.device)

        # Generate audio
        y_g_hat = self.generator(**code_dict)
        if isinstance(y_g_hat, tuple):
            y_g_hat = y_g_hat[0]

        # Post-process: denormalize and convert to proper format
        audio_out = y_g_hat.squeeze(0)  # Remove batch dimension
        if audio_out.dim() == 2:
            audio_out = audio_out.squeeze(0)  # Remove channel dimension if present

        # Ensure output is (1, T) for torchaudio compatibility
        if audio_out.dim() == 1:
            audio_out = audio_out.unsqueeze(0)

        # Resample if needed
        if out_sr != self.h.sampling_rate:
            audio_out = torchaudio.functional.resample(
                audio_out, self.h.sampling_rate, out_sr
            )

        audio_np = audio_out.cpu().squeeze().numpy()
        sf.write(output_file, audio_np, 16000)

    # ---------- Helper Methods ----------
    def _ensure_tensor(self, x: Any) -> torch.Tensor:
        """Convert input to torch.Tensor on correct device."""
        if isinstance(x, torch.Tensor):
            return x.to(self.device)
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).to(self.device)
        raise TypeError(f"Expected Tensor or ndarray; got {type(x)}")

    def _extract_content_codes(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract content codes from audio.

        For now, this creates dummy codes since HuBERT extraction requires additional
        checkpoints. In production, this should use HuBERT + K-means quantization.
        """
        # Calculate expected sequence length based on code_hop_size
        expected_len = len(audio) // self.h.code_hop_size

        # Option 1: Try to load HuBERT if available
        hubert_ckpt = self.checkpoints_dir / "hubert_base_ls960.pt"
        kmeans_ckpt = self.checkpoints_dir / "km.bin"

        if hubert_ckpt.exists() and kmeans_ckpt.exists():
            try:
                from fairseq_feature_reader import HubertFeatureReader
                import joblib

                if not hasattr(self, '_hubert_reader'):
                    self._hubert_reader = HubertFeatureReader(
                        checkpoint_path=str(hubert_ckpt),
                        layer=6,
                        max_chunk=1600000
                    )
                    # Suppress sklearn version warnings when loading k-means model
                    import warnings as warn_module
                    with warn_module.catch_warnings():
                        warn_module.filterwarnings('ignore', category=UserWarning, module='sklearn')
                        self._kmeans = joblib.load(str(kmeans_ckpt))
                    self._print("Loaded HuBERT and K-means models")

                # Save temp audio file for HuBERT
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                    temp_path = f.name
                    import soundfile as sf
                    sf.write(temp_path, audio, sr)

                # Extract features and quantize
                feats = self._hubert_reader.get_feats(temp_path)
                codes = self._kmeans.predict(feats.cpu().numpy())

                # Clean up
                Path(temp_path).unlink()

                return codes.astype(np.int64)

            except Exception as e:
                warnings.warn(f"HuBERT extraction failed: {e}. Using dummy codes.")

        # Option 2: Fallback to dummy codes (zeros)
        # This allows the wrapper to run without HuBERT but won't produce good results
        warnings.warn(
            "Using dummy content codes. For proper voice conversion, provide HuBERT "
            f"checkpoint at {hubert_ckpt} and K-means model at {kmeans_ckpt}"
        )
        return np.zeros(expected_len, dtype=np.int64)

    def _extract_f0(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract F0 contour from audio using YAAPT."""
        # get_yaapt_f0 expects a batch dimension, so add one if needed
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]  # (T,) -> (1, T)
        f0 = self._get_yaapt_f0(audio, rate=sr, interp=True)
        return f0
