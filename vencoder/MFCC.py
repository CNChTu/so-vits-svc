import numpy as np
import torch
from vencoder.dphubert.model import wav2vec2_model
from vencoder.encoder import SpeechEncoder
import torchaudio
import torch.nn.functional as F

class MFCC(SpeechEncoder):
    def __init__(self, vec_path="", device=None, sample_rate=44100, hop_length=512, win_length=2048):
        super().__init__()
        self.hidden_dim = 32
        self.win_length = win_length
        self.mfcc_transform = torchaudio.transforms.MFCC(sample_rate=sample_rate, n_mfcc=self.hidden_dim,
                                                    melkwargs={"n_fft": win_length, "hop_length": hop_length, "n_mels": 128, "center": False},
                                                    )

    def encoder(self, wav):
        feats = wav
        if len(feats.shape) == 2:  # double channels
            feats = feats.mean(0)
        assert len(feats.shape) == 1, feats.dim()
        feats = feats[None, ...]
        if isinstance(feats, np.ndarray):
            feats = torch.from_numpy(feats)
        feats = feats.to(torch.float32)
        if feats.shape[-1] % self.win_length != 0:
            padding = self.win_length - feats.shape[-1] % self.win_length
            padding = (padding//2, padding-padding//2)
            feats = F.pad(feats, padding)
            # feats = F.pad(feats, (0, self.win_length - feats.shape[-1])) 
        units = self.mfcc_transform(feats)
        return units
