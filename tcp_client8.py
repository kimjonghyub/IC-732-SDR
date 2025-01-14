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
BUTTON_WIDTH, BUTTON_HEIGHT = 100, 50
FPS = 15
BUTTON_AREA_HEIGHT = 50
FFT_GRAPH_HEIGHT = (HEIGHT - BUTTON_AREA_HEIGHT) // 2
WATERFALL_HEIGHT = HEIGHT - BUTTON_AREA_HEIGHT - FFT_GRAPH_HEIGHT

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("RTL-TCP IQ Data Visualization")
clock = pygame.time.Clock()

button_rect = pygame.Rect((10, 10), (BUTTON_WIDTH, BUTTON_HEIGHT))
button_color = (255, 0, 0)  # Initial color: red
button_text = "Connect"
font = pygame.font.SysFont(None, 24)

data_queue = asyncio.Queue(maxsize=5)
waterfall_buffer = np.zeros((WATERFALL_HEIGHT, WIDTH), dtype=np.uint8)
waterfall_surface = pygame.Surface((WIDTH, WATERFALL_HEIGHT))
COLORMAP = np.zeros((256, 3), dtype=np.uint8)

for i in range(256):
    if i < 128: 
        r = 0
        g = int(i * 2)       
        b = int(255 - i * 2) 
    else: 
        r = int((i - 128) * 2) 
        g = int(255 - (i - 128) * 2) 
        b = 0

    COLORMAP[i] = [r, g, b]


is_connected = False
reader = None
writer = None
tasks = []  

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
    y_coords = BUTTON_AREA_HEIGHT + FFT_GRAPH_HEIGHT - magnitudes
    points = np.column_stack((x_coords, y_coords)).astype(int)
    pygame.draw.rect(screen, (0, 0, 0), (0, BUTTON_AREA_HEIGHT, WIDTH, FFT_GRAPH_HEIGHT))  
    pygame.draw.lines(screen, (255, 255, 255), False, points.tolist(), 1)

async def send_command(writer, cmd_id, param):
    command = struct.pack(">BI", cmd_id, param)
    writer.write(command)
    await writer.drain()
    
async def receive_data(reader, queue):
    buffer = b""
    global is_connected
    try:
        while is_connected:
            data = await reader.read(BUFFER_SIZE)
            if not data:
                if is_connected:
                    print("No data received. Connection may be closed.")
                break
            buffer += data
            while len(buffer) >= BUFFER_SIZE:
                iq_data = buffer[:BUFFER_SIZE]
                buffer = buffer[BUFFER_SIZE:]
                if queue.full():
                    await queue.get()
                await queue.put(iq_data)
    except Exception as e:
        print(f"Error receiving data: {e}")
    finally:
        is_connected = False
        
async def handle_connection():
    global reader, writer, is_connected
    try:
        reader, writer = await asyncio.open_connection(SERVER_IP, SERVER_PORT)
        await send_command(writer, CMD_SET_SAMPLERATE, SAMPLE_RATE)
        await send_command(writer, CMD_SET_FREQ, FREQ)
        await send_command(writer, CMD_SET_AGC_MODE, AGC_MODE)
        is_connected = True
        print("Connected.")
        await asyncio.sleep(0.5)  
    except Exception as e:
        print(f"Connection failed: {e}")
        is_connected = False
        
async def close_connection():
    global reader, writer, is_connected, data_queue
    if writer:
        try:
            writer.close()
            await writer.wait_closed()
        except Exception as e:
            print(f"Error while disconnecting: {e}")
    reader, writer = None, None
    is_connected = False

    while not data_queue.empty():
        try:
            data_queue.get_nowait()
        except asyncio.QueueEmpty:
            break

    print("Disconnected.")

async def toggle_connection():
    global is_connected, tasks
    if is_connected:
        print("Disconnecting...")
        is_connected = False

        for task in tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        tasks.clear()

        await close_connection()
    else:
        print("Connecting...")
        await handle_connection()
        if is_connected:
            tasks.append(asyncio.create_task(receive_data(reader, data_queue)))
        
        
def draw_button():
    pygame.draw.rect(screen, button_color, button_rect)
    text = font.render(button_text, True, (255, 255, 255))
    screen.blit(text, (button_rect.x + (BUTTON_WIDTH - text.get_width()) // 2,
                       button_rect.y + (BUTTON_HEIGHT - text.get_height()) // 2))

async def main():
    global button_color, button_text, is_connected
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if button_rect.collidepoint(event.pos):
                    if is_connected:
                        button_color = (255, 0, 0)  
                        button_text = "Connect"
                        await toggle_connection()
                    else:
                        button_color = (0, 255, 0)  
                        button_text = "Disconnect"
                        await toggle_connection()

        screen.fill((0, 0, 0))

        if is_connected:
            try:
                iq_data = await data_queue.get()
                complex_signal = process_iq_data(iq_data)
                freqs, magnitudes = compute_fft(complex_signal, SAMPLE_RATE)
                scaled_magnitudes = np.interp(
                    magnitudes, [-50, 100], [0, 255]
                ).astype(np.uint8)
                update_waterfall(waterfall_buffer, scaled_magnitudes)
                draw_waterfall(waterfall_surface, waterfall_buffer)
                draw_fft_graph(screen, freqs, scaled_magnitudes)
            except asyncio.QueueEmpty:
                pass

        draw_button()
        pygame.display.flip()
        clock.tick(FPS)

    if is_connected:
        await toggle_connection()
    pygame.quit()

if __name__ == "__main__":
    asyncio.run(main())

