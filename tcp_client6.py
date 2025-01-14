import asyncio
import struct
import pygame
import numpy as np
import scipy.signal
from scipy.interpolate import make_interp_spline

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

SERVER_IP = "222.117.38.98"
SERVER_PORT = 1234
FREQ = 69011000
SAMPLE_RATE = 250000
GAIN = 405
FREQ_CORRECTION = 0
AGC_MODE = 1
DIRECT_SAMPLING = 0
GAIN_MODE = 0

WIDTH, HEIGHT = 400, 800
FPS = 15
WATERFALL_HEIGHT = 400
FFT_GRAPH_HEIGHT = HEIGHT - WATERFALL_HEIGHT

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("RTL-TCP IQ Data Visualization")
clock = pygame.time.Clock()

data_queue = asyncio.Queue(maxsize=5)

waterfall_buffer = np.zeros((WATERFALL_HEIGHT, WIDTH), dtype=np.uint8)
waterfall_surface = pygame.Surface((WIDTH, WATERFALL_HEIGHT))

COLORMAP = np.array([[i, 0, 255 - i] for i in range(256)], dtype=np.uint8)

def process_iq_data(iq_data):
    iq_data = np.frombuffer(iq_data, dtype=np.uint8).astype(np.float32)
    iq_data = (iq_data - 127.5) / 127.5
    real = iq_data[0::2]
    imag = iq_data[1::2]
    return real + 1j * imag

def compute_fft(complex_signal, sample_rate):
    window = scipy.signal.get_window("hann", len(complex_signal))
    windowed_signal = complex_signal * window
    fft_data = np.fft.fftshift(np.fft.fft(windowed_signal))
    fft_magnitude = 20 * np.log10(np.abs(fft_data) + 1e-6)
    freqs = np.fft.fftshift(np.fft.fftfreq(len(complex_signal), d=1/sample_rate))
    return freqs, fft_magnitude

def update_waterfall(buffer, fft_row):
    buffer[1:] = buffer[:-1]
    buffer[0] = np.interp(
        np.linspace(0, len(fft_row) - 1, WIDTH),
        np.arange(len(fft_row)),
        fft_row
    ).astype(np.uint8)

def draw_waterfall(surface, buffer):
    rgb_buffer = COLORMAP[buffer].reshape((WATERFALL_HEIGHT, WIDTH, 3))
    pygame.surfarray.blit_array(surface, np.transpose(rgb_buffer, (1, 0, 2)))
    screen.blit(surface, (0, HEIGHT - WATERFALL_HEIGHT))
    
def draw_fft_graph(screen, freqs, magnitudes):
    mid_x = WIDTH // 2
    x_coords = mid_x + (freqs / (SAMPLE_RATE / 2)) * mid_x
    y_coords = FFT_GRAPH_HEIGHT - magnitudes
    points = np.column_stack((x_coords, y_coords)).astype(int)
    pygame.draw.lines(screen, (255, 255, 255), False, points.tolist(), 1)

async def send_command(writer, cmd_id, param):
    command = struct.pack(">BI", cmd_id, param)
    writer.write(command)
    await writer.drain()
    
async def receive_data(reader, queue):
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
            await queue.put(iq_data)

async def process_data(queue):
    while True:
        iq_data = await queue.get()
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
    reader, writer = await asyncio.open_connection(SERVER_IP, SERVER_PORT)
    await send_command(writer, CMD_SET_SAMPLERATE, SAMPLE_RATE)
    await send_command(writer, CMD_SET_FREQ, FREQ)
    await send_command(writer, CMD_SET_AGC_MODE, AGC_MODE)
    try:
        await asyncio.gather(
            receive_data(reader, data_queue),
            process_data(data_queue),
        )
    finally:
        writer.close()
        await writer.wait_closed()
        pygame.quit()

if __name__ == "__main__":
    asyncio.run(main())
