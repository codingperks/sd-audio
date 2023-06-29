import torchaudio
import torchaudio.transforms as transforms
import IPython

class Processor:
    _mel_transform = None
        
    def update_transform(data, n_fft, n_mels, sample_rate):
        Processor._mel_transform = transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels
        )
        
    # Returns a spectrogram tensor from a waveform
    def wav_to_spec(data, n_fft, n_mels, sample_rate):            
            return Processor.mel_transform(data)

    # Converts spectrogram to wavform 
    def spec_to_wav(data, n_fft, sample_rate):
        n_stft = int((n_fft//2) + 1)
        
        inverse_transform = transforms.InverseMelScale(sample_rate=sample_rate, n_stft=n_stft)
        grifflim_transform = transforms.GriffinLim(n_fft=n_fft)

        mel_specgram = Processor.mel_transform(data)
        inverse_wav = inverse_transform(mel_specgram)
        psuedo_wav = grifflim_transform(inverse_wav)
        
        return psuedo_wav

    def play_wav(data, sample_rate):
        IPython.display.Audio(data.numpy(), rate=sample_rate)

    # Takes an existing dataset and creates a copy with a spectrogram columm - no prefix
    def dataset_to_spec(dataset, n_fft = 256, n_mels = 80):    
        transformed_data = []

        for data in dataset:
            waveform, sample_rate = torchaudio.load(data['audio']['path'], normalize=True)

            spectrogram = Processor.wav_to_spec(waveform, n_fft, n_mels, sample_rate)
            
            data['spectrogram'] = spectrogram
            transformed_data.append(data)
            
        return transformed_data

    # Takes an existing dataset and creates a copy with a spectrogram column, allows selection of data prefix
    def dataset_to_spec(dataset, prefix, n_fft = 256, n_mels = 80):
        prefix_len = len(prefix)
        filtered_data = [d for d in dataset if d["id"][:prefix_len] == prefix]
        
        transformed_data = []

        for data in filtered_data:
            waveform, sample_rate = torchaudio.load(data['audio']['path'], normalize=True)

            spectrogram = Processor.wav_to_spec(waveform, n_fft, n_mels, sample_rate)
            
            data['spectrogram'] = spectrogram
            transformed_data.append(data)
            
        return transformed_data
    
    
            
    """ 
    Transformation back into
    invers_transform = torchaudio.transforms.InverseMelScale(sample_rate=sample_rate, n_stft=n_stft)
    grifflim_transform = torchaudio.transforms.GriffinLim(n_fft=n_fft)

    mel_specgram = transform(waveform)
    inverse_waveform = invers_transform(mel_specgram)
    pseudo_waveform = grifflim_transform(inverse_waveform) """        