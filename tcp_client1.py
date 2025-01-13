import asyncio
import struct
import pygame
import numpy as np
import scipy.signal

# Constants for commands
MAGIC_HEADER = b"RTL0"
CMD_SET_FREQ = 0x01
CMD_SET_SAMPLERATE = 0x02
CMD_SET_GAIN = 0x04
CMD_SET_FREQ_CORRECTION = 0x05
CMD_SET_AGC_MODE = 0x08
CMD_SET_DIRECT_SAMPLING = 0x09
CMD_SET_GAIN_MODE = 0x03
CMD_SET_OFFEST_TURNING =  0x0a
FRAME_SIZE = 512
BUFFER_SIZE = FRAME_SIZE * 2

# Configuration
SERVER_IP = "192.168.10.216"
SERVER_PORT = 1234
FREQ = 69011000
SAMPLE_RATE = 250000
GAIN = 405  # Represented in tenths of dB (e.g., 40.0 dB -> 400)
FREQ_CORRECTION = 0
AGC_MODE = 1
DIRECT_SAMPLING = 0
GAIN_MODE = 0

# Pygame setup
WIDTH, HEIGHT = 400, 400
FPS = 30

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("RTL-TCP IQ Data Visualization")
clock = pygame.time.Clock()

# Queue for incoming IQ data
data_queue = asyncio.Queue(maxsize=1)

# Waterfall configuration
waterfall_height = 200
fft_graph_height = 200
waterfall_surface = pygame.Surface((WIDTH, waterfall_height))
waterfall_buffer = np.zeros((waterfall_height, WIDTH), dtype=np.uint8)

# Function to send commands to the server
async def send_command(writer, cmd_id, param):
    command = struct.pack(">BI", cmd_id, param)
    writer.write(command)
    await writer.drain()

# Function to convert raw IQ data to normalized complex signal
def process_iq_data(iq_data):
    iq_data = np.frombuffer(iq_data, dtype=np.uint8).astype(np.float32)
    iq_data = (iq_data - 127.5) / 127.5  # Normalize to range [-1, 1]
    real = iq_data[0::2]
    imag = iq_data[1::2]
    return real + 1j * imag  # Return complex signal

# Function to perform FFT on a subset of the signal
def compute_fft(complex_signal, sample_rate, frame_size=FRAME_SIZE):
    
    if len(complex_signal) < frame_size:
        return None, None

    sampled_signal = complex_signal[:frame_size]
    window = scipy.signal.get_window("hann", frame_size)
    windowed_signal = sampled_signal * window
    fft_data = np.fft.fftshift(np.fft.fft(windowed_signal, n=frame_size))
    fft_magnitude = np.abs(fft_data)
    fft_magnitude_db = 20 * np.log10(fft_magnitude + 1e-6)  # Convert to dB
    freqs = np.fft.fftshift(np.fft.fftfreq(frame_size, d=1 / sample_rate))
    return freqs, fft_magnitude_db

# Function to format frequency
def format_frequency(freq):
    return f"{freq:,}".replace(",", ".") + " Hz"

# Function to apply smoothing to FFT data
def smooth_fft(fft_magnitude, alpha=0.1):
    smoothed = np.copy(fft_magnitude)
    for i in range(1, len(fft_magnitude)):
        smoothed[i] = alpha * fft_magnitude[i] + (1 - alpha) * smoothed[i - 1]
    return smoothed

# Function to update waterfall
def update_waterfall(waterfall_buffer, fft_magnitude_scaled):
    waterfall_buffer[1:] = waterfall_buffer[:-1]
    waterfall_buffer[0] = np.interp(
        np.linspace(0, len(fft_magnitude_scaled) - 1, WIDTH),
        np.arange(len(fft_magnitude_scaled)),
        fft_magnitude_scaled
    ).astype(np.uint8)

# Function to draw the waterfall display
def draw_waterfall(surface, waterfall_buffer):
    for y in range(waterfall_buffer.shape[0]):
        for x in range(waterfall_buffer.shape[1]):
            color = waterfall_buffer[y, x]
            pygame.draw.line(surface, (color, color, color), (x, y), (x, y))
    screen.blit(surface, (0, HEIGHT - waterfall_height))

# Function to process IQ data, compute FFT, and draw it
def draw_panadapter(screen, iq_data):
    
    complex_signal = process_iq_data(iq_data)
    freqs, fft_magnitude_db = compute_fft(complex_signal, SAMPLE_RATE)
    fft_magnitude_db = smooth_fft(fft_magnitude_db, alpha=0.1)
    freqs = np.fft.fftshift(np.fft.fftfreq(len(complex_signal), d=1/SAMPLE_RATE))
    center_idx = len(freqs) // 2
    center_db = fft_magnitude_db[center_idx]

    # Display center frequency dB value and configured frequency
    font = pygame.font.Font(None, 24)
    center_db_text = font.render(f"Center Frequency Magnitude: {center_db:.2f} dB", True, (255, 255, 255))
    freq_text = font.render(f"Frequency: {format_frequency(FREQ)}", True, (255, 255, 255))

    screen.fill((0, 0, 0))
    screen.blit(center_db_text, (10, 10))
    screen.blit(freq_text, (310, 10))

    # Draw center frequency line
    mid_x = WIDTH // 2
    pygame.draw.line(screen, (255, 0, 0), (mid_x, 0), (mid_x, HEIGHT), 1)

    # Scale Y-axis to range [-50 dB, 100 dB]
    min_magnitude = -30
    max_magnitude = 30
    fft_magnitude_clipped = np.clip(fft_magnitude_db, min_magnitude, max_magnitude)
    fft_magnitude_scaled = ((fft_magnitude_clipped - min_magnitude) / (max_magnitude - min_magnitude) * 255).astype(np.uint8)

    # Update and draw waterfall
    update_waterfall(waterfall_buffer, fft_magnitude_scaled)
    draw_waterfall(waterfall_surface, waterfall_buffer)

    # Draw FFT data
    fft_magnitude_scaled_display = (fft_magnitude_clipped - min_magnitude) / (max_magnitude - min_magnitude) * fft_graph_height
    mid_x = WIDTH // 2
    points = []
    for i in range(len(freqs)):
        x = int(mid_x + freqs[i] / (SAMPLE_RATE / 2) * mid_x)
        y = HEIGHT - waterfall_height - int(fft_magnitude_scaled_display[i])
        points.append((x, y))

    pygame.draw.lines(screen, (0, 255, 0), False, points, 1)

    pygame.display.flip()

# Async task to receive data from the server
async def receive_data(reader, queue):
    try:
        buffer = b""
        while True:
            data = await reader.read(BUFFER_SIZE)
            if not data:
                print("Connection closed by server.")
                break
            buffer += data

            while len(buffer) >= BUFFER_SIZE:
                iq_data = buffer[:BUFFER_SIZE]
                buffer = buffer[BUFFER_SIZE:]
                if queue.full():
                    await queue.get()  
                await queue.put(iq_data)
    except asyncio.CancelledError:
        print("Data receiving cancelled.")
    except Exception as e:
        print(f"Error in receive_data: {e}")
        
# Async function to process and visualize data
async def process_data(queue):
    try:
        while True:
            if not queue.empty():
                iq_data = await queue.get()
                draw_panadapter(screen, iq_data)
            else:
                await asyncio.sleep(0.01)  
            clock.tick(FPS)
    except Exception as e:
        print(f"Error in process_data: {e}")


# Main function to connect and receive data
async def main():
    try:
        reader, writer = await asyncio.open_connection(SERVER_IP, SERVER_PORT)

        # Send configuration commands
        await send_command(writer, CMD_SET_SAMPLERATE, SAMPLE_RATE)
        await send_command(writer, CMD_SET_FREQ_CORRECTION, FREQ_CORRECTION)
        await send_command(writer, CMD_SET_FREQ, FREQ)
        await send_command(writer, CMD_SET_AGC_MODE, AGC_MODE)
        await send_command(writer, CMD_SET_DIRECT_SAMPLING, DIRECT_SAMPLING)
        await send_command(writer, CMD_SET_GAIN_MODE, GAIN_MODE)
        #await send_command(writer, CMD_SET_GAIN, GAIN)

        print("Commands sent to server. Receiving IQ data...")
        
        

         # Run receiver and processor concurrently
        await asyncio.gather(
            receive_data(reader, data_queue),
            process_data(data_queue),
        )



    except Exception as e:
        print(f"Error in main: {e}")

    finally:
        pygame.quit()

if __name__ == "__main__":
    asyncio.run(main())
