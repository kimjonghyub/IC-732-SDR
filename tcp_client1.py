import asyncio
import struct
import pygame
import numpy as np

# Constants for commands
MAGIC_HEADER = b"RTL0"
CMD_SET_FREQ = 0x01
CMD_SET_SAMPLERATE = 0x02
CMD_SET_GAIN = 0x04
CMD_SET_FREQ_CORRECTION = 0x05
CMD_SET_AGC_MODE = 0x08
CMD_SET_DIRECT_SAMPLING = 0x09
CMD_SET_GAIN_MODE = 0x03

# Configuration
SERVER_IP = "192.168.10.216"
SERVER_PORT = 1234
FREQ = 69011500
SAMPLE_RATE = 240000
GAIN = 405  # Represented in tenths of dB (e.g., 40.0 dB -> 400)
FREQ_CORRECTION = 0
AGC_MODE = 0
DIRECT_SAMPLING = 0
GAIN_MODE = 0

# Pygame setup
WIDTH, HEIGHT = 1024, 600
FPS = 30

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("RTL-TCP IQ Data Visualization")
clock = pygame.time.Clock()

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
    iq_data = np.frombuffer(iq_data, dtype=np.uint8).astype(np.float32)
    iq_data = (iq_data - 127.5) / 127.5  # Normalize to range [-1, 1]

    real = iq_data[0::2]
    imag = iq_data[1::2]
    complex_signal = real + 1j * imag

    # Perform FFT
    fft_data = np.fft.fftshift(np.fft.fft(complex_signal))
    fft_magnitude = np.abs(fft_data)
    fft_magnitude_db = 20 * np.log10(fft_magnitude + 1e-6)  # Convert to dB

    # Apply smoothing to FFT magnitude
    fft_magnitude_db = smooth_fft(fft_magnitude_db, alpha=0.2)

    # Define frequency axis (centered at 0 Hz)
    freqs = np.fft.fftshift(np.fft.fftfreq(len(complex_signal), d=1/SAMPLE_RATE))

    # Find center frequency magnitude
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
    #update_waterfall(waterfall_buffer, fft_magnitude_scaled)
    #draw_waterfall(waterfall_surface, waterfall_buffer)

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
        await send_command(writer, CMD_SET_GAIN, GAIN)

        print("Commands sent to server. Receiving IQ data...")

        running = True
        buffer = b""

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Read data from the server
            data = await reader.read(1024 * 1024)  # Adjust buffer size as needed
            if not data:
                break
            buffer += data

            # Process data if we have a full frame (e.g., 512 bytes per frame)
            frame_size = 6000
            while len(buffer) >= frame_size:
                iq_data = buffer[:frame_size]
                buffer = buffer[frame_size:]

                draw_panadapter(screen, iq_data)

            clock.tick(FPS)

    except Exception as e:
        print(f"Error: {e}")

    finally:
        pygame.quit()

if __name__ == "__main__":
    asyncio.run(main())
