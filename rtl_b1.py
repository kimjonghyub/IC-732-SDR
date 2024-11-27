from rtlsdr import RtlSdr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import RadioButtons
import matplotlib.gridspec as gridspec

# User-defined parameters
CENTER_FREQ = 69.011500e6  # Center frequency in Hz (69.0115 MHz)
SAMPLE_RATE = 2.56e6  # Sample rate in Hz
GAIN = 49.6  # Gain

# Bandwidth options in Hz
BANDWIDTH_OPTIONS = [200e3, 500e3, 960e3, 1.5e6]  # Bandwidth options

# Initialize the SDR
sdr = RtlSdr()
sdr.sample_rate = SAMPLE_RATE
sdr.center_freq = CENTER_FREQ
sdr.gain = GAIN

# FFT and plot setup
NFFT = 1024  # FFT size
history_size = 100  # Number of rows in the waterfall display
current_bandwidth = BANDWIDTH_OPTIONS[2]  # Default bandwidth (960 kHz)

# Create a figure with a GridSpec layout
fig = plt.figure(figsize=(10.24, 6), facecolor='black')
gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[1, 2], figure=fig)

# Line graph for PSD (Row 1, Col 1)
ax_line = fig.add_subplot(gs[0, 0], facecolor='black')
line, = ax_line.plot([], [], lw=1, color='cyan')
ax_line.set_xlabel('Frequency (MHz)', color='white')
ax_line.set_ylabel('Relative Power (dB)', color='white')
ax_line.set_title('Real-time Power Spectral Density', color='white')
ax_line.set_xlim(-0.5, 0.5)  # Adjust frequency range as per your data
ax_line.set_ylim(-50, 50)
ax_line.tick_params(axis='both', colors='white')
ax_line.grid(color='gray', linestyle='--')

# Make axes spines white
for spine in ax_line.spines.values():
    spine.set_edgecolor('white')

# Spectrogram (Waterfall) graph (Row 1, Col 2)
ax_waterfall = fig.add_subplot(gs[1, 0], facecolor='black')
waterfall_data = np.full((history_size, NFFT), -50)  # Default power level
waterfall = ax_waterfall.imshow(
    waterfall_data,
    aspect='auto',
    cmap='plasma',
    vmin=-50,
    vmax=50,
    origin='lower'
)
ax_waterfall.set_xlabel('Frequency (MHz)', color='white')
ax_waterfall.set_ylabel('Time (frames)', color='white')
ax_waterfall.set_title('Waterfall Spectrogram', color='white')
ax_waterfall.tick_params(axis='both', colors='white')

# Make axes spines white
for spine in ax_waterfall.spines.values():
    spine.set_edgecolor('white')

# Radio buttons for bandwidth selection (Row 2, Col 1)
ax_radio = fig.add_subplot(gs[0, 1], facecolor='black')
radio_buttons = RadioButtons(ax_radio, labels=[f"{bw / 1e3:.0f} kHz" for bw in BANDWIDTH_OPTIONS], activecolor='cyan')
for label in radio_buttons.labels:
    label.set_color('white')  # Set radio button labels to white
for circle in ax_radio.patches:
    circle.set_edgecolor('white')  # Set radio button markers to white

# Frequency axis and slicing for the selected range
def calculate_freqs(bandwidth):
    freqs_full = np.fft.fftshift(np.fft.fftfreq(NFFT, 1 / sdr.sample_rate))  # Full frequency bins in Hz
    freq_mask = (freqs_full >= -bandwidth / 2) & (freqs_full <= bandwidth / 2)  # Mask for selected range
    return freqs_full[freq_mask] / 1e6  # Return in MHz

freqs = calculate_freqs(current_bandwidth)

# Update the plots with live data
def update(frame):
    global waterfall_data, freqs
    samples = sdr.read_samples(8 * 1024)  # Collect samples
    fft_vals = np.fft.fft(samples, NFFT)  # Compute FFT
    fft_vals = np.fft.fftshift(fft_vals)  # Center FFT around 0
    psd_data = 10 * np.log10(np.abs(fft_vals) ** 2 + 1e-10)  # Convert to dB scale

    # Slice the PSD data to match the selected frequency range
    freqs_full = np.fft.fftshift(np.fft.fftfreq(NFFT, 1 / sdr.sample_rate))
    freq_mask = (freqs_full >= -current_bandwidth / 2) & (freqs_full <= current_bandwidth / 2)
    psd_data = psd_data[freq_mask]

    padded_psd_data = np.full(NFFT, -50)  
    padded_psd_data[:len(psd_data)] = psd_data  
    
    # Update the line plot
    line.set_data(freqs, psd_data)

    # Update the waterfall plot
    waterfall_data = np.roll(waterfall_data, -1, axis=0)  # Shift rows upward
    waterfall_data[-1, :] = padded_psd_data  # Insert the latest PSD data in the last row
    waterfall.set_data(waterfall_data)  # Update the image
    waterfall.set_extent([freqs[0], freqs[-1], 0, history_size])  # Adjust the extent for the new range
    return line, waterfall

# Bandwidth change handler
def change_bandwidth(label):
    global current_bandwidth, freqs, waterfall_data
    current_bandwidth = float(label)  # Update bandwidth
    freqs = calculate_freqs(current_bandwidth)  # Recalculate frequency range
    waterfall_data = np.full((history_size, len(freqs)), -50)  # Reset waterfall data
    ax_line.set_xlim(freqs[0], freqs[-1])  # Update x-axis limits
    waterfall.set_extent([freqs[0], freqs[-1], 0, history_size])  # Update waterfall extent

radio_buttons.on_clicked(lambda label: change_bandwidth(float(label.split()[0]) * 1e3))

# Animation with faster update intervals
ani = FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)

# Ensure SDR is closed when the plot window is closed
def on_close(event):
    sdr.close()

fig.canvas.mpl_connect('close_event', on_close)

plt.tight_layout()
plt.show()
