import torch

class VoiceControlWrapper:
    def __init__(self):
        raise NotImplementedError

    def extract_embedding(self, source_file: str) -> torch.Tensor:
        """Extract the speaker embedding from a source .wav file"""
        raise NotImplementedError

    def inference(self, source_file: str, output_file: str,
                  source_embedding: torch.Tensor, target_embedding: torch.Tensor):
        """Perform inference with a source file and target speaker embedding,
        writing to the output file"""
        raise NotImplementedError
