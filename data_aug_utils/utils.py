import random
import numpy as np
from .aug_fun import params2sos, change_gender, change_gender_f0
from scipy.signal import sosfilt

Qmin, Qmax = 2, 5
rng = np.random.default_rng()
Fc = np.exp(np.linspace(np.log(60), np.log(7600), 10))

def random_eq(wav, sr):
    z = rng.uniform(0, 1, size=(10,))
    Q = Qmin * (Qmax / Qmin)**z
    G = rng.uniform(-12, 12, size=(10,))
    sos = params2sos(G, Fc, Q, sr)
    wav = sosfilt(sos, wav)
    return wav

def random_formant_f0(wav, sr):
    #s = parselmouth.Sound(wav, sampling_frequency=sr)
    
    lo, hi = 60, 1100
    
    ratio_fs = rng.uniform(1, 1.4)
    coin = (rng.random() > 0.5)
    ratio_fs = coin*ratio_fs + (1-coin)*(1/ratio_fs)
    
    ratio_ps = rng.uniform(1, 2)
    coin = (rng.random() > 0.5)
    ratio_ps = coin*ratio_ps + (1-coin)*(1/ratio_ps)
    
    ratio_pr = rng.uniform(1, 1.5)
    coin = (rng.random() > 0.5)
    ratio_pr = coin*ratio_pr + (1-coin)*(1/ratio_pr)
    
    ss = change_gender(wav, sr, lo, hi, ratio_fs, ratio_ps, ratio_pr)
    
    return ss

def fixed_formant_f0(wav, sr):
    #s = parselmouth.Sound(wav, sampling_frequency=sr)
    lo, hi = 60, 1100
    
    ratio_fs, f0_med, ratio_pr = 0.8, 100, 0.8
        
    ss = change_gender_f0(wav, sr, lo, hi, ratio_fs, f0_med, ratio_pr)
    
    return ss  