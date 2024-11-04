import pyaudio

def check_supported_sample_rates(device_index):
    audio = pyaudio.PyAudio()
    sample_rates = [8000, 16000, 32000, 44100, 48000, 96000, 192000] 
    supported_rates = []
    
    for rate in sample_rates:
        try:
            if audio.is_format_supported(rate, input_device=device_index, input_channels=1, input_format=pyaudio.paInt16):
                supported_rates.append(rate)
        except ValueError:
            pass
    
    audio.terminate()
    return supported_rates

device_index = 1  
supported_rates = check_supported_sample_rates(device_index)
print(f"Device {device_index} supports the following sample rates: {supported_rates}")
