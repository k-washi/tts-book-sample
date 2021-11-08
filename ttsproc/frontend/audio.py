import torchaudio


def load_audio(path, sr):
    """
    srでpathの音ファイルを読み込む
    """
    data, raw_sr = torchaudio.load(path)
    data = torchaudio.transforms.Resample(raw_sr, sr)(data).squeeze(0)
    return data.cpu().detach().numpy()