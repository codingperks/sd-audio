import torchaudio
import torchaudio.transforms as transforms
import IPython


class Processor:
    _mel_transform = None
    _n_fft = None
    _n_mels = None
    _sample_rate = None

    def __init__(self, n_fft, n_mels, sample_rate):
        self._n_fft = n_fft
        self._n_mels = n_mels
        self._sample_rate = sample_rate
        self._mel_transform = transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, n_mels=n_mels
        )

    # Returns a spectrogram tensor from a waveform
    def wav_to_spec(self, data):
        return self._mel_transform(data)

    # Converts spectrogram to wavform
    def spec_to_wav(self, data):
        n_stft = int((self._n_fft // 2) + 1)

        inverse_transform = transforms.InverseMelScale(
            sample_rate=self._sample_rate, n_stft=n_stft
        )
        grifflim_transform = transforms.GriffinLim(n_fft=self._n_fft)

        mel_specgram = self._mel_transform(data)
        inverse_wav = inverse_transform(mel_specgram)
        psuedo_wav = grifflim_transform(inverse_wav)

        return psuedo_wav

    # Takes an existing dataset and creates a copy with a spectrogram columm - no prefix
    def dataset_to_spec(dataset, n_fft=256, n_mels=80):
        transformed_data = []

        for data in dataset:
            waveform, sample_rate = torchaudio.load(
                data["audio"]["path"], normalize=True
            )

            spectrogram = Processor.wav_to_spec(waveform, n_fft, n_mels, sample_rate)

            data["spectrogram"] = spectrogram
            transformed_data.append(data)

        return transformed_data

    # Takes an existing dataset and creates a copy with a spectrogram column, allows selection of data prefix
    def dataset_to_spec(dataset, prefix, n_fft=256, n_mels=80):
        prefix_len = len(prefix)
        filtered_data = [d for d in dataset if d["id"][:prefix_len] == prefix]

        transformed_data = []

        for data in filtered_data:
            waveform, sample_rate = torchaudio.load(
                data["audio"]["path"], normalize=True
            )

            spectrogram = Processor.wav_to_spec(waveform, n_fft, n_mels, sample_rate)

            data["spectrogram"] = spectrogram
            transformed_data.append(data)

        return transformed_data

    def play_wav(data, sample_rate):
        IPython.display.Audio(data.numpy(), rate=sample_rate)

    """ 
    Transformation back into
    invers_transform = torchaudio.transforms.InverseMelScale(sample_rate=sample_rate, n_stft=n_stft)
    grifflim_transform = torchaudio.transforms.GriffinLim(n_fft=n_fft)

    mel_specgram = transform(waveform)
    inverse_waveform = invers_transform(mel_specgram)
    pseudo_waveform = grifflim_transform(inverse_waveform) 
    """

    """     def update_transform(n_fft, n_mels, sample_rate):
    Processor._n_fft = n_fft
    Processor._n_mels = n_mels
    Processor._mel_transform = transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        n_mels=n_mels
    ) """
