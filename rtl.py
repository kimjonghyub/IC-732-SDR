from rtlsdr import RtlSdr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


# User-defined parameters
CENTER_FREQ = 69.011500e6  # Center frequency in Hz (69.0115 MHz)
SAMPLE_RATE = 2.56e6  # Sample rate in Hz (960 kHz)
GAIN = 49.6  # Gain
FREQ_RANGE = 960e3  # Frequency range in Hz (200 kHz bandwidth)

# Initialize the SDR
sdr = RtlSdr()
sdr.sample_rate = SAMPLE_RATE
sdr.center_freq = CENTER_FREQ
sdr.gain = GAIN

# FFT and plot setup
NFFT = 1024  # FFT size
history_size = 100  # Number of rows in the waterfall display

# Create a figure with two subplots
fig, (ax_line, ax_waterfall) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [1, 2]})

# Frequency axis and slicing for the selected range
freqs = np.fft.fftshift(np.fft.fftfreq(NFFT, 1 / sdr.sample_rate))  # Frequency bins in Hz
freq_mask = (freqs >= -FREQ_RANGE / 2) & (freqs <= FREQ_RANGE / 2)  # Mask for selected range
freqs = freqs[freq_mask] / 1e6  # Convert to MHz

# Line plot (PSD)
line, = ax_line.plot([], [], lw=2, color='blue')
ax_line.set_xlim(freqs[0], freqs[-1])  # Set x-axis limits
ax_line.set_ylim(-50, 50)  # Relative power in dB
ax_line.set_xlabel('Frequency (MHz)')
ax_line.set_ylabel('Relative Power (dB)')
ax_line.set_title('Real-time Power Spectral Density')

"""
# Annotation for the peak frequency
peak_annotation = ax_line.annotate(
    " ", xy=(0, 0), xytext=(-50, 50),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="-", color="red"),
    bbox=dict(boxstyle="round,pad=0.3", edgecolor="red", facecolor="white"),
    fontsize=10
)
"""
# Point to mark the peak
peak_point, = ax_line.plot([], [], 'ro', markersize=8, label='Peak Point')  # Red circle for the peak

# Waterfall plot (initialize with empty data matching the frequency range)
waterfall_data = np.full((history_size, len(freqs)), -120)  # Default power level
waterfall = ax_waterfall.imshow(
    waterfall_data,
    aspect='auto',
    extent=[freqs[0], freqs[-1], 0, history_size],
    cmap='plasma',
    vmin=-50,
    vmax=50,
    origin='lower'
)
ax_waterfall.set_xlabel('Frequency (MHz)')
ax_waterfall.set_ylabel('Time (frames)')
ax_waterfall.set_title('Waterfall Spectrogram')

# Data update function for animation
def update(frame):
    global waterfall_data
    samples = sdr.read_samples(8 * 1024)  # Collect samples
    fft_vals = np.fft.fft(samples, NFFT)  # Compute FFT
    fft_vals = np.fft.fftshift(fft_vals)  # Center FFT around 0
    psd_data = 10 * np.log10(np.abs(fft_vals) ** 2 + 1e-10)  # Convert to dB scale

    # Slice the PSD data to match the selected frequency range
    psd_data = psd_data[freq_mask]

     # Update the line plot
    line.set_data(freqs, psd_data)
    """
    # Find the peak frequency and update the annotation
    peak_idx = np.argmax(psd_data)  # Index of the peak
    peak_freq = freqs[peak_idx]  # Peak frequency in MHz
    peak_power = psd_data[peak_idx]  # Peak power in dB
    peak_annotation.set_text(f"{peak_freq:.3f} MHz")
    peak_annotation.set_position((freqs[peak_idx], peak_power))
    peak_annotation.xy = (peak_freq, peak_power)  # Move the annotation
    """
    # Find the peak frequency and update the marker
    peak_idx = np.argmax(psd_data)  # Index of the peak
    peak_freq = freqs[peak_idx]  # Peak frequency in MHz
    peak_power = psd_data[peak_idx]  # Peak power in dB
    peak_point.set_data([peak_freq], [peak_power])  # Update the peak point

    # Update the waterfall plot
    waterfall_data = np.roll(waterfall_data, -1, axis=0)  # Shift rows upward
    waterfall_data[-1, :] = psd_data  # Insert the latest PSD data in the last row
    waterfall.set_data(waterfall_data)  # Update the image
    return line, waterfall,  peak_point

# Animation with faster update intervals
ani = FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)  # 50ms interval (20 FPS)

# Ensure SDR is closed when the plot window is closed
def on_close(event):
    sdr.close()

fig.canvas.mpl_connect('close_event', on_close)

plt.tight_layout()
plt.show()
