# dpvc/controlvc.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, Any, Dict, Callable
import sys, importlib.util, types, tempfile, shutil
import numpy as np
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
        self.device = torch.device(
            device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
        )
        self.checkpoints_dir = (Path(checkpoints_dir).resolve()
                                if checkpoints_dir else (self.repo_root / "checkpoints"))
        self.config = config or {}

        if not self.repo_root.exists():
            raise FileNotFoundError(f"Control-VC repo root not found: {self.repo_root}")

        # Candidate script names Joe mentioned
        self._embed_candidates = [
            "infer_speaker_embedding.py",
            "infer_speakenbed.py",         # some forks typo this
            "inferspeakenbed.py",          # …and this
            "infer_speaker_embed.py",
        ]
        self._infer_candidates = [
            "infer_main.py",
            "inference.py",
            "infer.py",
        ]

        # Resolved modules (lazy)
        self._embed_mod: Optional[types.ModuleType] = None
        self._infer_mod: Optional[types.ModuleType] = None

        # Discovered callable hooks (lazy)
        self._extract_fn: Optional[Callable[..., torch.Tensor]] = None
        self._vc_infer_fn: Optional[Callable[..., torch.Tensor]] = None

    # ---------- public API ----------
    @torch.inference_mode()
    def extract_embedding(self, wav_path: Path) -> torch.Tensor:
        """
        Returns a speaker embedding (Torch tensor on self.device).
        Strategy:
          - Try to import a function-style API (e.g., extract_spk_embed(wav, sr, …))
          - Else, call module.main() programmatically into a temp dir and load .npy
        """
        wav_path = Path(wav_path).expanduser().resolve()
        if not wav_path.exists():
            raise FileNotFoundError(f"Reference wav not found: {wav_path}")

        self._ensure_embed_module()

        # 1) If the module exposes a proper function, use it.
        fn = self._extract_fn or self._find_extract_function(self._embed_mod)
        if fn:
            wav, sr = torchaudio.load(str(wav_path))
            wav = wav.to(self.device)
            out = fn(wav=wav, sr=sr, checkpoints=self.checkpoints_dir, device=str(self.device))
            return self._ensure_tensor(out)

        # 2) Fallback: call the script's main() with programmatic argv into temp dir.
        if hasattr(self._embed_mod, "main"):
            with tempfile.TemporaryDirectory(prefix="cv-embed-") as td:
                outdir = Path(td)
                # Common CLIs: --input / --wav / --wav_scp ; --out/--output
                argv = [
                    "infer_speaker_embedding.py",
                    "--input", str(wav_path),
                    "--output", str(outdir),
                    "--checkpoints", str(self.checkpoints_dir),
                ]
                self._call_module_main(self._embed_mod, argv)
                # Load first .npy produced
                npys = list(outdir.glob("*.npy"))
                if not npys:
                    raise RuntimeError("No embedding .npy produced by infer_speaker_embedding main().")
                emb = torch.from_numpy(np.load(npys[0])).to(self.device).float()
                return emb

        raise NotImplementedError(
            "Could not locate a callable embedding extractor in Control-VC. "
            "Looked for function names like: extract_spk_embed / get_spk_embed "
            "or a script main() in one of: "
            f"{self._embed_candidates}"
        )

    @torch.inference_mode()
    def infer(
        self,
        source_wav: Path,
        target_embedding: torch.Tensor,
        out_sr: int = 16000,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Runs VC using a precomputed (possibly DP-noised) speaker embedding.
        Strategy:
          - Try function-style API on infer module
          - Else programmatically call module.main() into a temp dir
        """
        source_wav = Path(source_wav).expanduser().resolve()
        if not source_wav.exists():
            raise FileNotFoundError(f"Source wav not found: {source_wav}")

        self._ensure_infer_module()
        target_embedding = self._ensure_tensor(target_embedding)

        # 1) If a clean function exists, use it.
        fn = self._vc_infer_fn or self._find_infer_function(self._infer_mod)
        if fn:
            wav, sr = torchaudio.load(str(source_wav))
            wav = wav.to(self.device)
            out = fn(
                src=wav, sr=sr, target_embedding=target_embedding,
                checkpoints=self.checkpoints_dir, device=str(self.device),
                out_sr=out_sr, **kwargs
            )
            return self._ensure_waveform(out, out_sr)

        # 2) Fallback: script main() path (write emb to temp file, call, read wav)
        if hasattr(self._infer_mod, "main"):
            with tempfile.TemporaryDirectory(prefix="cv-infer-") as td:
                tmp = Path(td)
                emb_path = tmp / "tgt_emb.npy"
                np.save(emb_path, target_embedding.detach().cpu().float().numpy())

                out_wav = tmp / "out.wav"
                argv = [
                    "infer_main.py",
                    "--source", str(source_wav),
                    "--embedding", str(emb_path),
                    "--output", str(out_wav),
                    "--checkpoints", str(self.checkpoints_dir),
                    "--sr", str(out_sr),
                ]
                self._call_module_main(self._infer_mod, argv)

                if not out_wav.exists():
                    # Some scripts dump to a folder; pick the first WAV we find.
                    candidates = list(tmp.rglob("*.wav"))
                    if not candidates:
                        raise RuntimeError("No waveform produced by infer_main main().")
                    out_wav = candidates[0]

                wav, sr = torchaudio.load(str(out_wav))
                if sr != out_sr:
                    wav = torchaudio.functional.resample(wav, sr, out_sr)
                return wav.to(self.device)

        raise NotImplementedError(
            "Could not locate a callable inference endpoint in Control-VC. "
            "Looked for functions like vc_infer / infer / convert, or a script main(). "
            f"Tried files: {self._infer_candidates}"
        )

    # ---------- helpers ----------
    def _ensure_tensor(self, x: Any) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.to(self.device)
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).to(self.device)
        raise TypeError(f"Expected Tensor or ndarray; got {type(x)}")

    def _ensure_waveform(self, x: Any, sr: int) -> torch.Tensor:
        t = self._ensure_tensor(x)
        # Expect [C, T] or [T]; normalize to [1, T]
        if t.ndim == 1:
            t = t.unsqueeze(0)
        return t

    def _call_module_main(self, mod: types.ModuleType, argv: list[str]) -> None:
        """Temporarily replace sys.argv and call module.main()"""
        old = sys.argv[:]
        try:
            sys.argv = ["prog"] + argv[1:] if argv and argv[0].endswith(".py") else argv
            mod.main()
        finally:
            sys.argv = old

    def _ensure_embed_module(self) -> None:
        if self._embed_mod is None:
            self._embed_mod = self._load_first_existing(self._embed_candidates)
        # pre-discover callable for speed (optional)
        self._extract_fn = self._find_extract_function(self._embed_mod)

    def _ensure_infer_module(self) -> None:
        if self._infer_mod is None:
            self._infer_mod = self._load_first_existing(self._infer_candidates)
        self._vc_infer_fn = self._find_infer_function(self._infer_mod)

    def _load_first_existing(self, filenames: list[str]) -> types.ModuleType:
        for name in filenames:
            fp = (self.repo_root / name)
            if fp.exists():
                return self._load_module_from_path(fp)
        # also search a bit deeper (1 level) if scripts live in subfolders
        for name in filenames:
            found = list(self.repo_root.rglob(name))
            if found:
                return self._load_module_from_path(found[0])
        raise FileNotFoundError(
            f"None of these Control-VC files were found under {self.repo_root}:\n"
            + "\n".join(f"  - {n}" for n in filenames)
        )

    def _load_module_from_path(self, path: Path) -> types.ModuleType:
        spec = importlib.util.spec_from_file_location(path.stem, str(path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import module from {path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[path.stem] = mod
        # Ensure repo_root on path so relative imports inside the script work
        sys.path.insert(0, str(self.repo_root))
        try:
            spec.loader.exec_module(mod)  # type: ignore[attr-defined]
        finally:
            # keep repo_root on sys.path for nested imports during lifetime
            pass
        return mod

    # Heuristic discovery of callable names inside Control-VC scripts
    def _find_extract_function(self, mod: types.ModuleType) -> Optional[Callable]:
        candidates = [
            "extract_spk_embed", "get_spk_embed", "extract_embedding",
            "infer_speaker_embedding", "extract_speaker_embedding"
        ]
        for name in candidates:
            fn = getattr(mod, name, None)
            if callable(fn):
                return fn
        return None

    def _find_infer_function(self, mod: types.ModuleType) -> Optional[Callable]:
        candidates = [
            "vc_infer", "infer", "convert", "run_inference", "inference",
        ]
        for name in candidates:
            fn = getattr(mod, name, None)
            if callable(fn):
                return fn
        return None