import pyaudio
import numpy as np
import plotly.graph_objects as go
from scipy.fft import fft
from scipy.ndimage import gaussian_filter1d

CHUNK = 1024
RATE = 44100
min_freq = 400
max_freq = 4000
min_amp = 0.1
max_amp = 7

# PyAudio setup
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=RATE,
                input=True,
                input_device_index=2,
                frames_per_buffer=CHUNK)

# Frequency and time setup for the waterfall plot
time_points = 45
x = np.arange(time_points)
freqs = np.fft.fftfreq(CHUNK, 1.0 / RATE)[:CHUNK // 2]
freq_indices = np.where((freqs >= min_freq) & (freqs <= max_freq))
filtered_freqs = freqs[freq_indices] / 1000
Z_data = np.zeros((len(filtered_freqs), time_points))

# Initial Plotly figure setup
fig = go.Figure()

fig.update_layout(
    title="3D Spectrum Waterfall",
    scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(title="Frequency (kHz)", tickvals=np.linspace(min_freq / 1000, max_freq / 1000, 10)),
        zaxis=dict(title="Amplitude", range=[min_amp * 1.e6, max_amp * 1.e6]),
        aspectratio=dict(x=1, y=1, z=0.5),
    ),
)

try:
    while True:
        data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
        fft_data = np.abs(fft(data)[:CHUNK // 2])
        Z = gaussian_filter1d(fft_data[freq_indices], sigma=1.5)
        Z_data = np.roll(Z_data, -1, axis=1)
        Z_data[:, -1] = Z

        X, Y = np.meshgrid(x, filtered_freqs)
        
        fig.update_traces(overwrite=True)
        fig.add_trace(go.Surface(
            z=Z_data,
            x=X,
            y=Y,
            colorscale="Plasma",
            cmin=min_amp * 1.e6,
            cmax=max_amp * 1.e6,
            showscale=False,
        ))

        fig.show()
        
except Exception as e:
    print(f"Error: {e}")
finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
    print("Audio stream closed.")
