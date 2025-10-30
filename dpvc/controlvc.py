# dpvc/controlvc.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, Any, Dict
import torch
import torchaudio

class ControlVCWrapper:
    """
    Control-VC wrapper with a two-stage API:
      1) extract_embedding(wav_path) -> torch.Tensor
      2) infer(source_wav, target_embedding, **kwargs) -> torch.Tensor (waveform)

    Notes:
    - This version removes subprocess usage per Joe's request.
    - It mirrors the OpenVoice wrapper's split (embedding vs inference).
    - Checkpoints are expected to be present locally (manual download).
    """

    def __init__(
        self,
        repo_root: Path,
        device: str = "cpu",
        checkpoints_dir: Optional[Path] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.repo_root = Path(repo_root).expanduser().resolve()
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.checkpoints_dir = Path(checkpoints_dir) if checkpoints_dir else self.repo_root / "checkpoints"
        self.config = config or {}

        # ---- TODO(phase-0.5): import Control-VC modules directly ----
        # Example (will vary by fork):
        # from control_vc.infer import load_models, extract_spk_embed, vc_infer
        # self._load_models = load_models
        # self._extract_spk_embed = extract_spk_embed
        # self._vc_infer = vc_infer
        #
        # For now, keep attributes as None and raise clear errors where needed.
        self._load_models = None
        self._extract_spk_embed = None
        self._vc_infer = None

        # Lazy-loaded model handles
        self._models_loaded = False
        self._handles = {}

    # ---------- lifecycle ----------
    def _ensure_models(self) -> None:
        if self._models_loaded:
            return
        # Validate checkpoints exist (manual for now, per Joe)
        if not self.checkpoints_dir.exists():
            raise FileNotFoundError(
                f"Control-VC checkpoints not found at {self.checkpoints_dir}. "
                f"Please download them manually (see Control-VC README)."
            )
        # If Control-VC exposes a loader, call it here:
        if self._load_models is None:
            # Leave a precise error so Joe/Ivy know exactly what’s still needed
            raise NotImplementedError(
                "Direct Control-VC API not wired yet: bind self._load_models/ _extract_spk_embed/ _vc_infer "
                "to the Control-VC Python functions. No subprocess allowed."
            )
        self._handles = self._load_models(self.checkpoints_dir, device=str(self.device), **self.config)
        self._models_loaded = True

    # ---------- public API ----------
    @torch.inference_mode()
    def extract_embedding(self, wav_path: Path) -> torch.Tensor:
        """
        Compute a speaker embedding from a single WAV file.
        Mirrors OpenVoice's extract_embedding().
        """
        self._ensure_models()
        if self._extract_spk_embed is None:
            raise NotImplementedError("Control-VC: `_extract_spk_embed` binding not set.")
        wav, sr = torchaudio.load(str(wav_path))
        wav = wav.to(self.device)
        return self._extract_spk_embed(wav, sr, **self._handles)

    @torch.inference_mode()
    def infer(
        self,
        source_wav: Path,
        target_embedding: torch.Tensor,
        out_sr: int = 16000,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Voice conversion using a target (possibly-noised) speaker embedding.
        This is the place the anonymizer will pass the DP-perturbed embedding later.
        """
        self._ensure_models()
        if self._vc_infer is None:
            raise NotImplementedError("Control-VC: `_vc_infer` binding not set.")
        src, sr = torchaudio.load(str(source_wav))
        src = src.to(self.device)
        wav = self._vc_infer(src, sr, target_embedding=target_embedding, out_sr=out_sr, **self._handles)
        return wav
