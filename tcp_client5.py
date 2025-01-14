import asyncio
import struct
import pygame
import numpy as np
import scipy.signal
from scipy.interpolate import make_interp_spline

# Constants for commands
MAGIC_HEADER = b"RTL0"
CMD_SET_FREQ = 0x01
CMD_SET_SAMPLERATE = 0x02
CMD_SET_GAIN = 0x04
CMD_SET_FREQ_CORRECTION = 0x05
CMD_SET_AGC_MODE = 0x08
CMD_SET_DIRECT_SAMPLING = 0x09
CMD_SET_GAIN_MODE = 0x03
CMD_SET_OFFSET_TUNING = 0x0A
FRAME_SIZE = 512
BUFFER_SIZE = FRAME_SIZE * 2

# Configuration
SERVER_IP = "222.117.38.98"
SERVER_PORT = 1234
FREQ = 69011000
SAMPLE_RATE = 250000
GAIN = 405  # Represented in tenths of dB (e.g., 40.5 dB -> 405)
FREQ_CORRECTION = 0
AGC_MODE = 1
DIRECT_SAMPLING = 0
GAIN_MODE = 0

# Pygame setup
WIDTH, HEIGHT = 600, 400
FPS = 30
WATERFALL_HEIGHT = 200
FFT_GRAPH_HEIGHT = HEIGHT - WATERFALL_HEIGHT

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("RTL-TCP IQ Data Visualization")
clock = pygame.time.Clock()

# Queue for incoming IQ data
data_queue = asyncio.Queue(maxsize=5)

# Waterfall buffer
waterfall_buffer = np.zeros((WATERFALL_HEIGHT, WIDTH), dtype=np.uint8)
waterfall_surface = pygame.Surface((WIDTH, WATERFALL_HEIGHT))

# Utility functions
def process_iq_data(iq_data):
    print("Processing IQ data...")
    iq_data = np.frombuffer(iq_data, dtype=np.uint8).astype(np.float32)
    iq_data = (iq_data - 127.5) / 127.5  # Normalize to range [-1, 1]
    real = iq_data[0::2]
    imag = iq_data[1::2]
    return real + 1j * imag

def compute_fft(complex_signal, sample_rate):
    print(f"Computing FFT for {len(complex_signal)} samples...")
    window = scipy.signal.get_window("hann", len(complex_signal))
    windowed_signal = complex_signal * window
    fft_data = np.fft.fftshift(np.fft.fft(windowed_signal))
    fft_magnitude = 20 * np.log10(np.abs(fft_data) + 1e-6)  # Convert to dB
    freqs = np.fft.fftshift(np.fft.fftfreq(len(complex_signal), d=1/sample_rate))
    return freqs, fft_magnitude

def update_waterfall(buffer, fft_row):
    print("Updating waterfall...")
    buffer[1:] = buffer[:-1]
    buffer[0] = np.interp(
        np.linspace(0, len(fft_row) - 1, WIDTH),
        np.arange(len(fft_row)),
        fft_row
    ).astype(np.uint8)

def draw_waterfall(surface, buffer):
    print("Drawing waterfall...")
    print(f"Waterfall buffer shape: {buffer.shape}, Surface size: {surface.get_size()}")

    # Convert buffer to RGB
    rgb_buffer = np.stack((buffer,) * 3, axis=-1)

    # Get surface dimensions (width, height)
    surface_width, surface_height = surface.get_size()

    # Ensure buffer dimensions match the surface size (height, width for numpy)
    if rgb_buffer.shape[0] != surface_height or rgb_buffer.shape[1] != surface_width:
        print(f"Resizing buffer: Current {rgb_buffer.shape[:2]}, Target {(surface_height, surface_width)}")
        resized_buffer = np.zeros((surface_height, surface_width, 3), dtype=np.uint8)
        
        # Resize each channel independently
        for i in range(3):  # Resize R, G, B channels
            resized_buffer[..., i] = np.interp(
                np.linspace(0, rgb_buffer.shape[1] - 1, surface_width),  # Width
                np.arange(rgb_buffer.shape[1]),
                np.interp(
                    np.linspace(0, rgb_buffer.shape[0] - 1, surface_height),  # Height
                    np.arange(rgb_buffer.shape[0]),
                    rgb_buffer[..., i]
                )
            )
        rgb_buffer = resized_buffer

    # Debugging: Verify final buffer size
    print(f"Final buffer shape: {rgb_buffer.shape}, Surface size: {surface.get_size()}")

    # Blit the resized buffer to the surface
    try:
        surface_array = pygame.surfarray.pixels3d(surface)  # Access pixel array of the surface
        np.copyto(surface_array, np.transpose(rgb_buffer, (1, 0, 2)))  # Match width and height dimensions
        del surface_array  # Release lock on the surface
        screen.blit(surface, (0, HEIGHT - WATERFALL_HEIGHT))
    except ValueError as e:
        print(f"Error blitting array to surface: {e}")
        print(f"Buffer shape: {rgb_buffer.shape}, Surface size: {surface.get_size()}")
        raise





    
def draw_fft_graph(screen, freqs, magnitudes):
    print("Drawing FFT graph...")
    mid_x = WIDTH // 2
    x_coords = mid_x + (freqs / (SAMPLE_RATE / 2)) * mid_x
    y_coords = FFT_GRAPH_HEIGHT - magnitudes
    points = np.column_stack((x_coords, y_coords)).astype(int)
    pygame.draw.lines(screen, (255, 255, 255), False, points.tolist(), 1)

async def send_command(writer, cmd_id, param):
    print(f"Sending command {cmd_id} with parameter {param}...")
    command = struct.pack(">BI", cmd_id, param)
    writer.write(command)
    await writer.drain()

async def receive_data(reader, queue):
    print("Receiving data from server...")
    buffer = b""
    while True:
        data = await reader.read(BUFFER_SIZE)
        if not data:
            print("No data received. Connection may be closed.")
            break
        buffer += data

        while len(buffer) >= BUFFER_SIZE:
            iq_data = buffer[:BUFFER_SIZE]
            buffer = buffer[BUFFER_SIZE:]
            if queue.full():
                await queue.get()
            print("Adding data to queue...")
            await queue.put(iq_data)
            
async def process_data(queue):
    print("Starting data processing...")
    while True:
        iq_data = await queue.get()
        print("Processing a new IQ data frame...")
        complex_signal = process_iq_data(iq_data)
        freqs, magnitudes = compute_fft(complex_signal, SAMPLE_RATE)

        scaled_magnitudes = np.interp(
            magnitudes, [-50, 100], [0, 255]
        ).astype(np.uint8)
        update_waterfall(waterfall_buffer, scaled_magnitudes)
        screen.fill((0, 0, 0))
        draw_waterfall(waterfall_surface, waterfall_buffer)
        draw_fft_graph(screen, freqs, scaled_magnitudes)
        pygame.display.flip()
        clock.tick(FPS)

async def main():
    print("Connecting to server...")
    reader, writer = await asyncio.open_connection(SERVER_IP, SERVER_PORT)
    print("Connected to server.")
    await send_command(writer, CMD_SET_SAMPLERATE, SAMPLE_RATE)
    await send_command(writer, CMD_SET_FREQ, FREQ)
    await send_command(writer, CMD_SET_AGC_MODE, AGC_MODE)
    
    try:
        print("Starting main tasks...")
        await asyncio.gather(
            receive_data(reader, data_queue),
            process_data(data_queue),
        )
    finally:
        print("Closing connection and quitting...")
        writer.close()
        await writer.wait_closed()
        pygame.quit()

if __name__ == "__main__":
    print("Starting application...")
    asyncio.run(main())
