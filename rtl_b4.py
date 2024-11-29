import pyaudio
from rtlsdr import RtlSdr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import RadioButtons
import matplotlib.gridspec as gridspec
import Hamlib
import threading
import time
import pygame

# Global variables for Pygame and threading
stop_event = threading.Event()
current_freq = "Initializing..."

# User-defined parameters
CENTER_FREQ = 69.011500e6  # Center frequency in Hz (69.0115 MHz)
SAMPLE_RATE = 2.56e6  # Sample rate in Hz
GAIN = 49.6  # Gain
CHUNK = 512  # Audio buffer size
RATE = 44100  # Audio sample rate

# FFT and plot setup
NFFT = 512  # FFT size
history_size = 100  # Number of rows in the waterfall display

# Bandwidth options in Hz
BANDWIDTH_OPTIONS = [200e3, 500e3, 960e3, 1.5e6]  # Bandwidth options
current_bandwidth = BANDWIDTH_OPTIONS[2]  # Default bandwidth (960 kHz)

# Audio bandwidth options in Hz
AUDIO_BANDWIDTH_OPTIONS = [RATE // 8, RATE // 6, RATE // 4, RATE // 2]  # Example: 5512 Hz, 11025 Hz, 22050 Hz
current_audio_bandwidth = AUDIO_BANDWIDTH_OPTIONS[2]  # Default to maximum bandwidth


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
def pygame_overlay():
    """Pygame overlay for displaying the rig frequency."""
    global pygame_running, current_freq

    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((400, 200))
    pygame.display.set_caption("Rig Frequency Overlay")
    font = pygame.font.Font(None, 36)
    background_color = (0, 0, 0)  # Black
    text_color = (0, 255, 255)  # Cyan

    # Main Pygame loop
    while not stop_event.is_set():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame_running = False

        # Clear the screen
        screen.fill(background_color)

        # Render the frequency text
        text_surface = font.render(current_freq, True, text_color)
        screen.blit(text_surface, (50, 80))  # Position the text

        # Update the Pygame display
        pygame.display.flip()

        time.sleep(0.1)

    # Cleanup Pygame
    pygame.quit()
    
def init_plot():
    global fig, line, peak_point, waterfall, waterfall_data, audio_fft_line, ax_line, ax_waterfall, ax_radio, ax_audio_radio, ax_audio_fft, ax_rig_freq, rig_freq_text
    # Create a figure with subplots and widgets
    fig = plt.figure(figsize=(8, 4.8), facecolor='black')
    fig.canvas.manager.set_window_title("3D Spectrum Waterfall")  
    
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
 
# Frequency axis and slicing for the selected range
def calculate_freqs(bandwidth):
    
    freqs_full = np.fft.fftshift(np.fft.fftfreq(NFFT, 1 / SAMPLE_RATE))  # Full frequency bins in Hz
    freq_mask = (freqs_full >= -bandwidth / 2) & (freqs_full <= bandwidth / 2)  # Mask for selected range
    
    return freqs_full[freq_mask] / 1e6  # Return in MHz

freqs = calculate_freqs(current_bandwidth)

"""
def update_rig_frequency():
    
    global rig_freq_text, rig, stop_thread, 
    while not stop_thread:
        try:
            rig_freq = rig.get_freq()  # Get frequency from rig
            current_freq = f"{rig_freq / 1e6:.6f} MHz"  # Format the frequency
            if rig_freq_text:
                rig_freq_text.set_text(current_freq)
        except Exception as e:
            print(f"Hamlib error in frequency thread: {e}")
            if rig_freq_text:
                rig_freq_text.set_text("Error reading frequency")
        time.sleep(0.1)  # Adjust polling interval as needed (e.g., 500ms)
"""    

def update_rig_frequency():
    """Thread function to periodically update the rig frequency."""
    init_hamlib()
    global current_freq, rig, pygame_running
    while not stop_event.is_set():
        try:
            freq = rig.get_freq()  # Get frequency from rig
            current_freq = f"Frequency: {freq / 1e6:.6f} MHz"  # Format the frequency
        except Exception as e:
            current_freq = "Error reading frequency"
            print(f"Hamlib error: {e}")
        time.sleep(0.1)  # Adjust polling interval as needed

# Data update function for animation
def update_graph():
    global waterfall_data, freqs
    while not stop_event.is_set():
        try:
    
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
            #plt.grid(True)

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
            
            fig.canvas.draw_idle()
        except Exception as e:
            print(f"Error updating graphs: {e}")
        time.sleep(0.05)
        
    

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


def main():
    global line, waterfall, audio_fft_line, waterfall_data, stop_event
    
    # Initialize components
    
    init_sdr()
    init_audio()
    init_plot()
 
    
    graph_thread = threading.Thread(target=update_graph, daemon=True)
    rig_thread = threading.Thread(target=update_rig_frequency, daemon=True)
    pygame_thread = threading.Thread(target=pygame_overlay, daemon=True)
    
    graph_thread.start()
    rig_thread.start()
    pygame_thread.start()
    
    
    # Add radio buttons for bandwidth selection
    radio_buttons = RadioButtons(ax_radio, labels=[f"{bw / 1e3:.0f} kHz" for bw in BANDWIDTH_OPTIONS],activecolor='cyan')
    radio_buttons.on_clicked(lambda label: change_bandwidth(float(label.split()[0]) * 1e3))
    for label in radio_buttons.labels:  # Change label color to white
        label.set_color('white')  # Set the text color of radio button labels to white
    for circle in ax_radio.patches:  # Radio button markers
        circle.set_edgecolor('white')  # White edges for buttons
    

    # Add radio buttons for audio bandwidth selection
    audio_radio_buttons = RadioButtons(ax_audio_radio, labels=[f"{bw} Hz" for bw in AUDIO_BANDWIDTH_OPTIONS], activecolor='cyan')
    audio_radio_buttons.on_clicked(change_audio_bandwidth)
    # Set label colors to white
    for label in audio_radio_buttons.labels:
        label.set_color('white')
    
    try:
        plt.tight_layout()
        plt.show()
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        stop_event.set()
        rig_thread.join()
        pygame_thread.join()
        graph_thread.join()

        # Cleanup
        stream.stop_stream()
        stream.close()
        p.terminate()
        rig.close()
        sdr.close()
        pygame.quit()
 
if __name__ == "__main__":
    main()

