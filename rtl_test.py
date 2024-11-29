import pyaudio
from rtlsdr import RtlSdr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import RadioButtons
from matplotlib.widgets import Button
from matplotlib.ticker import FuncFormatter
import matplotlib.gridspec as gridspec
import Hamlib
import threading
import time

current_freq = "Initializing..."

# User-defined parameters
CENTER_FREQ = 69.011500e6  # Center frequency in Hz (69.0115 MHz)
SAMPLE_RATE = 2.56e6  # Sample rate in Hz
GAIN = 49.6  # Gain
CHUNK = 512  # Audio buffer size
RATE = 44100  # Audio sample rate

# FFT and plot setup
NFFT = 1024  # FFT size
history_size = 100  # Number of rows in the waterfall display

# Bandwidth options in Hz
BANDWIDTH_OPTIONS = [200e3, 500e3, 960e3, 1.5e6, 2.56e6]  # Bandwidth options
current_bandwidth = BANDWIDTH_OPTIONS[2]  # Default bandwidth (960 kHz)
current_bandwidth_idx = 2  # Index to track current bandwidth
current_bandwidth = BANDWIDTH_OPTIONS[current_bandwidth_idx]

# Audio bandwidth options in Hz
AUDIO_BANDWIDTH_OPTIONS = [4000, 8000, 12000, 16000, 22050]  # Example: 5512 Hz, 7,350Hz, 11025 Hz, 22050 Hz
current_audio_bandwidth = AUDIO_BANDWIDTH_OPTIONS[2]  # Default to maximum bandwidth
current_audio_bandwidth_idx = 1  # Index to track current bandwidth
current_audio_bandwidth = AUDIO_BANDWIDTH_OPTIONS[current_audio_bandwidth_idx]


def init_hamlib():
    global rig
    Hamlib.rig_set_debug(Hamlib.RIG_DEBUG_NONE)
    rig = Hamlib.Rig(3021)  # Replace with your specific rig model
    rig.set_conf("rig_pathname", "/dev/ttyUSB0")
    rig.set_conf("serial_speed", "9600")
    rig.open()
    
def init_sdr():
    global sdr
    sdr = RtlSdr()
    sdr.sample_rate = SAMPLE_RATE
    sdr.center_freq = CENTER_FREQ
    sdr.gain = GAIN
    
def init_audio():
    global stream, p
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=RATE,
        input=True,
        input_device_index=2,
        frames_per_buffer=CHUNK,
    )

def init_plot():
    global fig, line, peak_point, waterfall, waterfall_data, audio_fft_line, ax_line, ax_waterfall, ax_radio, ax_audio_radio, ax_audio_fft, ax_rig_freq, rig_freq_text,freqs
    freqs = calculate_freqs(current_bandwidth)
    # Create a figure with subplots and widgets
    fig = plt.figure(figsize=(8, 4.8), facecolor='black')
    fig.canvas.manager.set_window_title("3D Spectrum Waterfall")  
    
    # Create GridSpec layout
    gs = gridspec.GridSpec(3, 3, width_ratios=[3, 0.2, 0.2], height_ratios=[1,0.1,2])
    
    # Assign subplots
    ax_line = fig.add_subplot(gs[0:2, 0], facecolor='black')  # Line graph
    ax_waterfall = fig.add_subplot(gs[2, 0], facecolor='black')  # Waterfall
    ax_audio_fft = fig.add_subplot(gs[2, 1:], facecolor='black')  # Audio FFT
    ax_rig_freq = fig.add_subplot(gs[1, 1:], facecolor='black')  # Adjust GridSpec as needed
    
    ax_line.set_position([0.05, 0.6, 0.6, 0.30])  # [left, bottom, width, height]
    ax_waterfall.set_position([0.05, 0.1, 0.6, 0.5])  # [left, bottom, width, height]
    ax_audio_fft.set_position([0.71, 0.1, 0.25, 0.5])  # [left, bottom, width, height]
    ax_rig_freq.set_position([0.7, 0.6, 0.25, 0.35])  # [left, bottom, width, height]

    # Line plot 
    line, = ax_line.plot([], [], lw=1, color='blue')
    ax_line.set_xlim(freqs[0], freqs[-1])  # Set x-axis limits
    ax_line.set_ylim(-100, 100)  # Relative power in dB
    ax_line.set_xlabel('Frequency (MHz)',color='white')
    ax_line.set_ylabel('Relative Power (dB)',color='white')
    ax_line.set_title('Real-time Power Spectral Density',color='white')
    ax_line.set_xticks([])
    ax_line.tick_params(axis='x', colors='black')  # White tick marks
    ax_line.tick_params(axis='y', colors='white')  # White tick marks
    ax_line.grid(color='gray', linestyle='--')  # Gray grid lines
    
    # Make axes (spines) white
    for spine in ax_line.spines.values():
        spine.set_edgecolor('white')  # Set axes color to white

    # Point to mark the peak
    peak_point, = ax_line.plot([], [], 'ro', markersize=8, label='Peak Point')  # Red circle for the peak

    # Waterfall plot 
    waterfall_data = np.full((history_size, len(freqs)), -120)  # Default power level
    waterfall = ax_waterfall.imshow(
        waterfall_data,
        aspect='auto',
        extent=[freqs[0], freqs[-1], 0, history_size],
        cmap='ocean',
        vmin=-100,
        vmax=100,
        origin='lower'
    )
    ax_waterfall.set_xlabel('Frequency (MHz)',color='white')
    ax_waterfall.set_ylabel('Time (frames)',color='white')
    ax_waterfall.tick_params(axis='both', colors='white')  # White tick marks
    ax_waterfall.set_yticks([])
    
    # Make axes (spines) white
    for spine in ax_waterfall.spines.values():
        spine.set_edgecolor('white')  # Set axes color to white
    
    # FFT for audio input
    ax_audio_fft.set_title('Audio FFT', color='white')
    audio_fft_line, = ax_audio_fft.plot([], [], lw=0.5, color='green')
    ax_audio_fft.set_xlim(0, (current_audio_bandwidth/1e3))
    ax_audio_fft.set_ylim(0, 150)
    ax_audio_fft.set_xlabel('Frequency (kHz)', color='white')
    ax_audio_fft.tick_params(axis='both', colors='white')
    ax_audio_fft.grid(color='gray', linestyle='--')

    for spine in ax_audio_fft.spines.values():
        spine.set_edgecolor('white')
    
    # RIG FREQ plot 
    ax_rig_freq.set_title('Rig Frequency', color='white', fontsize=12)
    ax_rig_freq.axis('off')  # Turn off axis lines for clean display
    rig_freq_text = ax_rig_freq.text(
        0.5, 0.5, '', color='cyan', fontsize=15, ha='center', va='center', transform=ax_rig_freq.transAxes
    )

def calculate_integer_ticks(bandwidth, num_ticks):
    max_tick = int(np.ceil(bandwidth / 1e3))  
    step = max(1, max_tick // (num_ticks - 1))  
    return np.arange(0, max_tick + 1, step)  

def update_audio_ticks(ax, bandwidth, num_ticks):
    ticks = calculate_integer_ticks(bandwidth, num_ticks)  
    ax.set_xticks(ticks)  
    ax.set_xlim(0, ticks[-1])  

def update_waterfall_ticks(ax, bandwidth, num_ticks):
    max_freq = bandwidth / 1e6  # Bandwidth in MHz
    ticks = np.linspace(-max_freq / 2, max_freq / 2, num_ticks)  # Centered frequency ticks
    ax.set_xticks(ticks)  # Set x-axis ticks
    ax.set_xlim(ticks[0], ticks[-1])  # Set x-axis limits
  
def waterfall_tick_label(value, tick_number):
    
    global rig_freq
    return f"{(value * 1e6 + rig_freq) / 1e6:.3f}"  


def calculate_freqs(bandwidth):
    
    freqs_full = np.fft.fftshift(np.fft.fftfreq(NFFT, 1 / SAMPLE_RATE))  # Full frequency bins in Hz
    freq_mask = (freqs_full >= -bandwidth / 2) & (freqs_full <= bandwidth / 2)  # Mask for selected range
    
    return freqs_full[freq_mask] / 1e6  # Return in MHz

#freqs = calculate_freqs(current_bandwidth)

def update_rig_frequency():
    global rig_freq_text, rig, stop_thread, rig_freq
    init_hamlib()
    while not stop_thread:
        try:
            rig_freq = rig.get_freq()  # Get frequency from rig
            current_freq = f"{rig_freq:,.0f}".replace(",", ".") + " MHz"
            if rig_freq_text:
                rig_freq_text.set_text(current_freq)
        except Exception as e:
            print(f"Hamlib error in frequency thread: {e}")
            if rig_freq_text:
                rig_freq_text.set_text("Error reading frequency")
        time.sleep(0.1)  # Adjust polling interval as needed (e.g., 500ms)

def update(frame):
    global waterfall_data, freqs
    
    samples = sdr.read_samples(4 * 1024)  # Collect samples
    fft_vals = np.fft.fft(samples, NFFT)  # Compute FFT
    fft_vals = np.fft.fftshift(fft_vals)  # Center FFT around 0
    psd_data = 10 * np.log10(np.abs(fft_vals) ** 2 + 1e-10)  # Convert to dB scale
    
    # Slice the PSD data to match the selected frequency range
    freqs_full = np.fft.fftshift(np.fft.fftfreq(NFFT, 1 / sdr.sample_rate))
    freq_mask = (freqs_full >= -current_bandwidth / 2) & (freqs_full <= current_bandwidth / 2)
    psd_data = psd_data[freq_mask]

    # Update the line plot
    line.set_data(freqs, psd_data)
    update_waterfall_ticks(ax_line, current_bandwidth, num_ticks=5)
    
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
    ax_waterfall.xaxis.set_major_formatter(FuncFormatter(waterfall_tick_label))
    update_waterfall_ticks(ax_waterfall, current_bandwidth, num_ticks=5)
    
    # Audio FFT
    audio_data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
    audio_fft_vals = np.fft.rfft(audio_data)
    audio_psd_data = 10 * np.log10(np.abs(audio_fft_vals) ** 2 + 1e-10)
    freqs_audio = np.fft.rfftfreq(len(audio_data), 1 / RATE) / 1e3
    audio_fft_line.set_data(freqs_audio, audio_psd_data)
    update_audio_ticks(ax_audio_fft, current_audio_bandwidth, num_ticks=5)
    
    # Apply bandwidth filter
    valid_indices = freqs_audio <= (current_audio_bandwidth / 1e3)
    audio_fft_line.set_data(freqs_audio[valid_indices], audio_psd_data[valid_indices])
    return line, waterfall, peak_point, audio_fft_line, rig_freq_text

def change_bandwidth(label):
    global current_bandwidth, freqs, waterfall_data
    current_bandwidth = float(label)  # Update bandwidth
    freqs = calculate_freqs(current_bandwidth)  # Recalculate frequency range
    waterfall_data = np.full((history_size, len(freqs)), -50)  # Reset waterfall data
    ax_line.set_xlim(freqs[0], freqs[-1])  # Update x-axis limits
    waterfall.set_extent([freqs[0], freqs[-1], 0, history_size])  # Update waterfall extent

def update_sdr_bandwidth(event):
    global current_bandwidth_idx, current_bandwidth
    
    current_bandwidth_idx = (current_bandwidth_idx + 1) % len(BANDWIDTH_OPTIONS)
    current_bandwidth = BANDWIDTH_OPTIONS[current_bandwidth_idx]
    change_bandwidth(current_bandwidth)
    
def update_af_bandwidth(event):
    global current_audio_bandwidth_idx, current_audio_bandwidth
    
    current_audio_bandwidth_idx = (current_audio_bandwidth_idx + 1) % len(AUDIO_BANDWIDTH_OPTIONS)
    current_audio_bandwidth = AUDIO_BANDWIDTH_OPTIONS[current_audio_bandwidth_idx]
    change_audio_bandwidth(current_audio_bandwidth)
 
def change_audio_bandwidth(label):
    global current_audio_bandwidth, ax_audio_fft
    current_audio_bandwidth = float(label)
    ax_audio_fft.set_xlim(0, current_audio_bandwidth / 1e3)  # Update X-axis range


def main():
    global line, waterfall, audio_fft_line, waterfall_data, stop_thread
    
    
    # Initialize components
    init_sdr()
    init_audio()
    init_plot()
    
    # Start rig frequency update thread
    stop_thread = False
    rig_thread = threading.Thread(target=update_rig_frequency, daemon=True)
    rig_thread.start()
    
    button_sdr_ax = plt.axes([0.052, 0.105, 0.1, 0.075])  # [left, bottom, width, height]
    sdr_bandwidth_button = Button(button_sdr_ax, "SDR B/W", color="lightgrey", hovercolor="grey")
    sdr_bandwidth_button.on_clicked(update_sdr_bandwidth)
    
    button_audio_ax = plt.axes([0.548, 0.105, 0.1, 0.075])  # [left, bottom, width, height]
    af_bandwidth_button = Button(button_audio_ax, "AF B/W", color="lightgrey", hovercolor="grey")
    af_bandwidth_button.on_clicked(update_af_bandwidth)

    # Animation
    ani = FuncAnimation(fig, update, interval=100, cache_frame_data=False)
    fig.canvas.mpl_connect("close_event", lambda _: stop_threads())
    
    #plt.tight_layout()
    plt.show()

# Ensure SDR is closed when the plot window is closed
def stop_threads():
    """Stop background threads and cleanup resources."""
    global stop_thread
    stop_thread = True
    try:
        stream.stop_stream()
        stream.close()
        p.terminate()
        sdr.close()
        rig.close()
    except Exception as e:
        print(f"Error during cleanup: {e}")


if __name__ == "__main__":
    main()

