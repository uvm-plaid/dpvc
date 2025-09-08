# Writing a Voice Control Wrapper

To implement a wrapper for a new voice control system, write a class
definition that extends `VoiceControlWrapper`. This class should
implement an `extract_embedding` method to extract speaker embeddings,
and an `inference` method to perform inference. A template appears
below:

``` py
class MyVoiceControlWrapper(VoiceControlWrapper):
    def __init__(self):
        # Put code here to load model checkpoints and initialize the VC system

    def extract_embedding(self, source_file: str) -> torch.Tensor:
        """Extract the speaker embedding from a source .wav file"""
        # Put code here to extract a single speaker embedding from a source file

    def inference(self, source_file: str, output_file: str,
                  source_embedding: torch.Tensor, target_embedding: torch.Tensor):
        """Perform inference with a source file and target speaker embedding,
        writing to the output file"""
        # Put code here to perform inference
```

After defining this class, an instance of it can be used to construct
an `Anonymizer` and perform differentially private anonymization in
the same way as previous examples.
