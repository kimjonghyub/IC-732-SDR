import pyaudio
from rtlsdr import RtlSdr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import RadioButtons
import matplotlib.gridspec as gridspec
import Hamlib

# Initialize Hamlib
Hamlib.rig_set_debug(Hamlib.RIG_DEBUG_NONE)  # Set Hamlib debug level
rig = Hamlib.Rig(3021)  # Replace with your specific rig model
rig.set_conf ("rig_pathname", "/dev/ttyUSB0")  # Adjust to your rig's device path
rig.set_conf ("serial_speed", "9600")  # Baudrate
rig.open()  # Open connection to the rig

# User-defined parameters
CENTER_FREQ = 69.011500e6  # Center frequency in Hz (69.0115 MHz)
SAMPLE_RATE = 2.56e6  # Sample rate in Hz
GAIN = 49.6  # Gain
CHUNK = 1024  # Audio buffer size
RATE = 44100  # Audio sample rate

# Bandwidth options in Hz
BANDWIDTH_OPTIONS = [200e3, 500e3, 960e3, 1.5e6]  # Bandwidth options

# Audio bandwidth options in Hz
AUDIO_BANDWIDTH_OPTIONS = [RATE // 8, RATE // 6, RATE // 4, RATE // 2]  # Example: 5512 Hz, 11025 Hz, 22050 Hz
current_audio_bandwidth = AUDIO_BANDWIDTH_OPTIONS[2]  # Default to maximum bandwidth


p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=RATE,
                input=True,
                input_device_index=2, 
                frames_per_buffer=CHUNK)

# Initialize the SDR
sdr = RtlSdr()
sdr.sample_rate = SAMPLE_RATE
sdr.center_freq = CENTER_FREQ
sdr.gain = GAIN

# FFT and plot setup
NFFT = 1024  # FFT size
history_size = 100  # Number of rows in the waterfall display
current_bandwidth = BANDWIDTH_OPTIONS[2]  # Default bandwidth (960 kHz)

# Create a figure with subplots and widgets
fig = plt.figure(figsize=(8, 4.8), facecolor='black')
#ax_line = fig.add_subplot(221,facecolor='black')
#ax_waterfall = fig.add_subplot(223,facecolor='black')
#ax_radio = fig.add_axes(222,[0.8, 0.5, 0.10, 0.2],facecolor='black')  # Position for the radio buttons
fig.canvas.manager.set_window_title("3D Spectrum Waterfall")  
"""
gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[1, 2])
ax_line = fig.add_subplot(gs[0, 0], facecolor='black')
ax_waterfall = fig.add_subplot(gs[1, 0], facecolor='black')
ax_radio = fig.add_subplot(gs[0, 1], facecolor='black')
ax_audio_fft = fig.add_subplot(gs[1, 1], facecolor='black')  # FFT for audio input
ax_radio.set_title('SDR Band Width', color='white')
ax_audio_fft.set_title('Audio FFT', color='white')
ax_audio_radio = fig.add_subplot(gs[0, 1], facecolor='black')  # Add new axis for audio radio buttons
"""
# Create GridSpec layout
gs = gridspec.GridSpec(3, 3, width_ratios=[3, 0.2, 0.2], height_ratios=[1,0.1,2])

# Assign subplots
ax_line = fig.add_subplot(gs[0:2, 0], facecolor='black')  # Line graph
ax_waterfall = fig.add_subplot(gs[2, 0], facecolor='black')  # Waterfall
ax_radio = fig.add_subplot(gs[0, 1], facecolor='black')  # SDR Band Width
ax_audio_radio = fig.add_subplot(gs[0, 2], facecolor='black')  # Audio Band Width
ax_audio_fft = fig.add_subplot(gs[2, 1:], facecolor='black')  # Audio FFT

# Titles for each section
ax_radio.set_title('SDR B/W', color='white',fontsize=10)
ax_audio_radio.set_title('Audio B/W', color='white',fontsize=10)
ax_audio_fft.set_title('Audio FFT', color='white')

# Add subplot for displaying radio frequency
ax_rig_freq = fig.add_subplot(gs[1, 1:], facecolor='black')  # Adjust GridSpec as needed
ax_rig_freq.set_title('Rig Frequency', color='white', fontsize=10, pad=10)
ax_rig_freq.axis('off')  # Turn off axis lines for clean display
rig_freq_text = ax_rig_freq.text(
    0.5, 0.5, '', color='cyan', fontsize=12, ha='center', va='center', transform=ax_rig_freq.transAxes
)


# Frequency axis and slicing for the selected range
def calculate_freqs(bandwidth):
    freqs_full = np.fft.fftshift(np.fft.fftfreq(NFFT, 1 / sdr.sample_rate))  # Full frequency bins in Hz
    freq_mask = (freqs_full >= -bandwidth / 2) & (freqs_full <= bandwidth / 2)  # Mask for selected range
    return freqs_full[freq_mask] / 1e6  # Return in MHz

freqs = calculate_freqs(current_bandwidth)

# Line plot (PSD)
line, = ax_line.plot([], [], lw=1, color='blue')
ax_line.set_xlim(freqs[0], freqs[-1])  # Set x-axis limits
ax_line.set_ylim(-50, 50)  # Relative power in dB
ax_line.set_xlabel('Frequency (MHz)',color='white')
ax_line.set_ylabel('Relative Power (dB)',color='white')
ax_line.set_title('Real-time Power Spectral Density',color='white')
ax_line.tick_params(axis='both', colors='white')  # White tick marks
ax_line.grid(color='gray', linestyle='--')  # Gray grid lines

# Make axes (spines) white
for spine in ax_line.spines.values():
    spine.set_edgecolor('white')  # Set axes color to white

# Point to mark the peak
peak_point, = ax_line.plot([], [], 'ro', markersize=8, label='Peak Point')  # Red circle for the peak

# Waterfall plot (initialize with empty data matching the frequency range)
waterfall_data = np.full((history_size, len(freqs)), -120)  # Default power level
waterfall = ax_waterfall.imshow(
    waterfall_data,
    aspect='auto',
    extent=[freqs[0], freqs[-1], 0, history_size],
    cmap='ocean',
    vmin=-50,
    vmax=50,
    origin='lower'
)
ax_waterfall.set_xlabel('Frequency (MHz)',color='white')
ax_waterfall.set_ylabel('Time (frames)',color='white')
ax_waterfall.set_title('Waterfall Spectrogram',color='white')
ax_waterfall.tick_params(axis='both', colors='white')  # White tick marks

# Make axes (spines) white
for spine in ax_waterfall.spines.values():
    spine.set_edgecolor('white')  # Set axes color to white
    
# FFT for audio input
audio_fft_line, = ax_audio_fft.plot([], [], lw=0.5, color='green')
ax_audio_fft.set_xlim(0, RATE // 6)
ax_audio_fft.set_ylim(-10, 200)
ax_audio_fft.set_xlabel('Frequency (Hz)', color='white')
ax_audio_fft.set_ylabel('Power (dB)', color='white')
ax_audio_fft.tick_params(axis='both', colors='white')
ax_audio_fft.grid(color='gray', linestyle='--')

for spine in ax_audio_fft.spines.values():
    spine.set_edgecolor('white')
    

# Data update function for animation
def update(frame):
    global waterfall_data, freqs
    samples = sdr.read_samples(24 * 1024)  # Collect samples
    fft_vals = np.fft.fft(samples, NFFT)  # Compute FFT
    fft_vals = np.fft.fftshift(fft_vals)  # Center FFT around 0
    psd_data = 10 * np.log10(np.abs(fft_vals) ** 2 + 1e-10)  # Convert to dB scale

    # Slice the PSD data to match the selected frequency range
    freqs_full = np.fft.fftshift(np.fft.fftfreq(NFFT, 1 / sdr.sample_rate))
    freq_mask = (freqs_full >= -current_bandwidth / 2) & (freqs_full <= current_bandwidth / 2)
    psd_data = psd_data[freq_mask]

    # Update the line plot
    line.set_data(freqs, psd_data)
    plt.grid(True)

    # Find the peak frequency and update the marker
    peak_idx = np.argmax(psd_data)  # Index of the peak
    peak_freq = freqs[peak_idx]  # Peak frequency in MHz
    peak_power = psd_data[peak_idx]  # Peak power in dB
    peak_point.set_data([peak_freq], [peak_power])  # Update the peak point

    # Update the waterfall plot
    waterfall_data = np.roll(waterfall_data, -1, axis=0)  # Shift rows upward
    waterfall_data[-1, :] = psd_data  # Insert the latest PSD data in the last row
    waterfall.set_data(waterfall_data)  # Update the image
    waterfall.set_extent([freqs[0], freqs[-1], 0, history_size])  # Adjust the extent for the new range
    
    # Audio FFT
    audio_data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
    audio_fft_vals = np.fft.rfft(audio_data)
    audio_psd_data = 10 * np.log10(np.abs(audio_fft_vals) ** 2 + 1e-10)
    freqs_audio = np.fft.rfftfreq(len(audio_data), 1 / RATE)
    audio_fft_line.set_data(freqs_audio, audio_psd_data)
    
    # Apply bandwidth filter
    valid_indices = freqs_audio <= current_audio_bandwidth
    audio_fft_line.set_data(freqs_audio[valid_indices], audio_psd_data[valid_indices])

    # Update rig frequency
    try:
        rig_freq = rig.get_freq()
        rig_freq_text.set_text(f"{rig_freq / 1e6:.6f} MHz")  # Display frequency in MHz
    except Exception as e:
        rig_freq_text.set_text("Error reading frequency")
        print(f"Hamlib error: {e}")
    
    return line, waterfall, peak_point, audio_fft_line, rig_freq_text

# Bandwidth change handler
def change_bandwidth(label):
    global current_bandwidth, freqs, waterfall_data
    current_bandwidth = float(label)  # Update bandwidth
    freqs = calculate_freqs(current_bandwidth)  # Recalculate frequency range
    waterfall_data = np.full((history_size, len(freqs)), -50)  # Reset waterfall data
    ax_line.set_xlim(freqs[0], freqs[-1])  # Update x-axis limits
    waterfall.set_extent([freqs[0], freqs[-1], 0, history_size])  # Update waterfall extent

# Audio bandwidth change handler
def change_audio_bandwidth(label):
    global current_audio_bandwidth, ax_audio_fft

    # Parse bandwidth from the label
    current_audio_bandwidth = int(label.split()[0])  # Extract the bandwidth value
    ax_audio_fft.set_xlim(0, current_audio_bandwidth)  # Update X-axis range


# Add radio buttons for bandwidth selection
radio_buttons = RadioButtons(ax_radio, labels=[f"{bw / 1e3:.0f} kHz" for bw in BANDWIDTH_OPTIONS],activecolor='cyan')
for label in radio_buttons.labels:  # Change label color to white
    label.set_color('white')  # Set the text color of radio button labels to white
for circle in ax_radio.patches:  # Radio button markers
    circle.set_edgecolor('white')  # White edges for buttons
radio_buttons.on_clicked(lambda label: change_bandwidth(float(label.split()[0]) * 1e3))

# Add radio buttons for audio bandwidth selection

audio_radio_buttons = RadioButtons(ax_audio_radio, labels=[f"{bw} Hz" for bw in AUDIO_BANDWIDTH_OPTIONS], activecolor='cyan')

# Set label colors to white
for label in audio_radio_buttons.labels:
    label.set_color('white')

# Connect the handler
audio_radio_buttons.on_clicked(change_audio_bandwidth)


# Animation with faster update intervals
ani = FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)  # 50ms interval (20 FPS)

# Ensure SDR is closed when the plot window is closed
def on_close(event):
    try:
        stream.stop_stream()
        stream.close()
        p.terminate()
        sdr.close()
        rig.close()  # Close the rig connection
    except Exception as e:
        print(f"Error during cleanup: {e}")


fig.canvas.mpl_connect('close_event', on_close)

plt.tight_layout()
plt.show()
