import pyopenjtalk
import numpy as np
import librosa
from ttsproc.frontend.dsp import logmelspectrogram, mulaw_quantize
from ttsproc.frontend.audio import load_audio
from ttsproc.frontend.openjtalk import pp_symbol, text_to_sequence
from ttsproc.util import pad_1d

FIX_HOP_SEC = 0.0125

def preprocess(wav_file, text, sr, mu, trim_space=0.05, trim_audio_db=10):
    """
    frame_shiftが0.0125で固定
    """
    # full context label estimate
    labels = pyopenjtalk.extract_fullcontext(text)
    PP = pp_symbol(labels)
    in_feats = np.array(text_to_sequence(PP), dtype=np.int64)

    _fix_hop_frame = int(sr * FIX_HOP_SEC)
    x = load_audio(wav_file, sr)
    if x.dtype in [np.int16, np.int32]:
        x = (x / np.iinfo(x.dtype).max).astype(np.float64)
    
    out_feats = logmelspectrogram(x, sr)

    x, index = librosa.effects.trim(x, top_db=trim_audio_db)
    start_frame = int(max(index[0] - trim_space * sr, 0) / _fix_hop_frame)
    end_frame = min((index[1] + trim_space * sr) // _fix_hop_frame, len(out_feats))

    x = x[int(start_frame * _fix_hop_frame):]
    out_feats = out_feats[start_frame:end_frame]
    length = _fix_hop_frame * out_feats.shape[0]
    x = pad_1d(x, length) if len(x) < length else x[:length]

    # 特徴量のアップサンプリングを行う都合上、音声波形の長さはフレームシフトで割り切れる必要があります
    assert len(x) % int(sr * FIX_HOP_SEC) == 0

    x = mulaw_quantize(x)
    return x, out_feats, in_feats, PP

    

    




