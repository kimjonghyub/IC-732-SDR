import asyncio
import struct
import pygame
import numpy as np
import scipy.signal
from scipy.interpolate import make_interp_spline

BLACK =    (  0,   0,   0)
WHITE =    (255, 255, 255)
GREEN =    (  0, 255,   0)
BLUE =     (  0,   0, 255)
RED =      (255,   0,   0)
YELLOW =   (192, 192,   0)
DARK_RED = (128,   0,   0)
LITE_RED = (255, 100, 100)
BGCOLOR =  (255, 230, 200)
BLUE_GRAY= (100, 100, 180)
ORANGE =   (255, 150,   0)
GRAY =     ( 60,  60,  60)
SCREEN =   (254, 165,   0)

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
BUFFER_SIZE = FRAME_SIZE * 6
FREQ_RANGE = 256

SERVER_IP = "222.117.38.98"
SERVER_PORT = 1234
FREQ = 69011000
SAMPLE_RATE = 1024000
GAIN = 405
FREQ_CORRECTION = 0
AGC_MODE = 1
DIRECT_SAMPLING = 0
GAIN_MODE = 0

WIDTH, HEIGHT = 400, 800
BUTTON_WIDTH, BUTTON_HEIGHT = 100, 35
FPS = 15
BUTTON_AREA_HEIGHT = 150
MARGIN = 50
FFT_GRAPH_HEIGHT = (HEIGHT - BUTTON_AREA_HEIGHT) // 2
WATERFALL_HEIGHT = HEIGHT - BUTTON_AREA_HEIGHT - FFT_GRAPH_HEIGHT
MIN_DB = -50
MAX_DB = 120

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("RTL-TCP IQ Data Visualization")
clock = pygame.time.Clock()

data_queue = asyncio.Queue(maxsize=2)

waterfall_buffer = np.zeros((WATERFALL_HEIGHT, WIDTH-MARGIN), dtype=np.uint8)
waterfall_surface = pygame.Surface((WIDTH-MARGIN, WATERFALL_HEIGHT))
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

ctl_box = pygame.Surface(((WIDTH), BUTTON_AREA_HEIGHT) )
ctl_box.fill(BLACK)
screen.blit(ctl_box, (5,5))

button_rect = pygame.Rect((10, 10), (BUTTON_WIDTH, BUTTON_HEIGHT))
button_color = (RED)  # Initial color: red
button_text = "Connect"
text_color = (BLACK)
font = pygame.font.SysFont(None, 24)
font_tick = pygame.font.SysFont(None, 20)

def text_objects(text, text_font, color):
    textSurface = font.render(text, 1, color)
    return textSurface, textSurface.get_rect()

def button(msg,x,y,w,h,ic,ac,tc,hc,action=None):
    mouse = pygame.mouse.get_pos()
    click = pygame.mouse.get_pressed()

    if x+w > mouse[0] > x and y+h > mouse[1] > y:
        pygame.draw.rect(screen, ac,(x,y,w,h))
        text_color = hc
        if click[0] == 1 and action != None:
            text_color = tc
            asyncio.create_task(action())
    else:
        pygame.draw.rect(screen, ic,(x,y,w,h))
        text_color = tc
    
    textSurf, textRect = text_objects(msg, font, text_color)
    textRect.center = ( (int(x)+(int(w/2))), (int(y)+(int(h/2))) )
    screen.blit(textSurf, textRect)

def upsample_signal(data, factor):
    x = np.arange(len(data))
    x_up = np.linspace(0, len(data) - 1, len(data) * factor)
    spline = make_interp_spline(x, data)
    return spline(x_up)

def process_iq_data(iq_data):
    iq_data = np.frombuffer(iq_data, dtype=np.uint8).astype(np.float32)
    iq_data = (iq_data - 127.5) / 127.5
    real = iq_data[0::2]
    imag = iq_data[1::2]
    return real + 1j * imag

async def compute_fft(complex_signal, sample_rate):
    window = scipy.signal.get_window("hann", len(complex_signal))
    windowed_signal = complex_signal * window
    fft_data = np.fft.fftshift(np.fft.fft(windowed_signal))
    fft_magnitude = 20 * np.log10(np.abs(fft_data) + 1e-6)
    freqs = np.fft.fftshift(np.fft.fftfreq(len(complex_signal), d=1/sample_rate))
    return freqs, fft_magnitude

def update_waterfall(buffer, fft_row):
    buffer[1:] = buffer[:-1]
    buffer[0] = np.interp(
        np.linspace(0, len(fft_row) - 1, WIDTH-MARGIN),
        np.arange(len(fft_row)),
        fft_row
    ).astype(np.uint8)

def draw_waterfall(surface, buffer):
    rgb_buffer = COLORMAP[buffer].reshape((WATERFALL_HEIGHT, WIDTH-MARGIN, 3))
    pygame.surfarray.blit_array(surface, np.transpose(rgb_buffer, (1, 0, 2)))
    screen.blit(surface, (MARGIN/2, HEIGHT - WATERFALL_HEIGHT))
    
def draw_fft_graph(screen, freqs, magnitudes):
    # Define reduced width and height for the graph
    graph_width = WIDTH - MARGIN
    graph_height = FFT_GRAPH_HEIGHT - MARGIN

    # Define the position of the graph
    graph_x = MARGIN/2
    graph_y = BUTTON_AREA_HEIGHT + MARGIN/2
    
    # Clear the FFT graph area
    pygame.draw.rect(screen, GRAY, (graph_x, graph_y, graph_width, graph_height))  
    
    mid_x = graph_x + graph_width // 2
    x_coords = mid_x + (freqs / (SAMPLE_RATE / 2)) * (graph_width // 2)
    """
    y_coords = graph_y + graph_height - 10 - (
        magnitudes / np.max(magnitudes) * (graph_height - 20)
    )
    """
    y_coords = graph_y + (
        graph_height * (1 - (magnitudes) / (graph_height))
    )
    y_coords = np.clip(y_coords, graph_y, graph_y + graph_height)
    points = np.column_stack((x_coords, y_coords)).astype(int)
    # Create polygon points to fill below the graph
    polygon_points = [(graph_x, graph_y + graph_height)] + points.tolist() + [
        (graph_x + graph_width, graph_y + graph_height)
    ]

    # Fill the area below the FFT graph with a color
    pygame.draw.polygon(screen, BLUE_GRAY, polygon_points)
    pygame.draw.lines(screen, (WHITE), False, points.tolist(), 1)
    
    # Draw x-axis
    pygame.draw.line(
        screen, GRAY, 
        (graph_x, graph_y + graph_height - 1), 
        (graph_x + graph_width, graph_y + graph_height - 1), 
        1
    )
    # Adjust the frequency range based on freq_range
    max_freq = SAMPLE_RATE / 2 / (1024 // FREQ_RANGE)  # Adjusted maximum frequency
    num_ticks = 4  # Number of ticks
    tick_spacing = graph_width // num_ticks  # Pixel spacing between ticks
    
    for i in range(num_ticks + 1):
        x_pos = i * tick_spacing
        freq = int((i - num_ticks // 2) * (max_freq / num_ticks) / 1e3)  # Frequency in kHz
        tick_label = f"{freq}kHz"
        
        # Draw tick line
        pygame.draw.line(
            screen, GRAY, 
            (x_pos+graph_x, graph_y + graph_height - 5), 
            (x_pos+graph_x, graph_y + graph_height + 5), 
            1
        )
        
        # Render tick label
        label = font_tick.render(tick_label, True, WHITE)
        label_rect = label.get_rect(center=(x_pos+graph_x, graph_y + graph_height + 15))
        screen.blit(label, label_rect)

async def set_freq_range(freq_range):
    global FREQ_RANGE, tasks
    FREQ_RANGE = freq_range
    if writer:
        try:
            await send_command(writer, CMD_SET_SAMPLERATE, SAMPLE_RATE)
            #await send_command(writer, CMD_SET_FREQ_CORRECTION, FREQ_CORRECTION)
            #await send_command(writer, CMD_SET_FREQ, FREQ)
            await send_command(writer, CMD_SET_AGC_MODE, AGC_MODE)
            await send_command(writer, CMD_SET_DIRECT_SAMPLING, DIRECT_SAMPLING)
            await send_command(writer, CMD_SET_GAIN_MODE, GAIN_MODE)
            
            #print(f"set {SAMPLE_RATE}")
        except Exception as e:
            print(f"set fail: {e}")
    print(f"set freq range : {FREQ_RANGE}kHz")

def process_signal(complex_signal, freq_range):
    global FPS, BUFFER_SIZE
    if freq_range == 256:
        complex_signal = complex_signal[::4]  
         
    elif freq_range == 512:
        complex_signal = complex_signal[::2]  
    
    elif freq_range == 1024:
        complex_signal = complex_signal[::1]  
         
        
    return complex_signal

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
        await send_command(writer, CMD_SET_FREQ_CORRECTION, FREQ_CORRECTION)
        await send_command(writer, CMD_SET_FREQ, FREQ)
        await send_command(writer, CMD_SET_AGC_MODE, AGC_MODE)
        await send_command(writer, CMD_SET_DIRECT_SAMPLING, DIRECT_SAMPLING)
        await send_command(writer, CMD_SET_GAIN_MODE, GAIN_MODE)
        #await send_command(writer, CMD_SET_GAIN, GAIN)
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
    """
    pygame.draw.rect(screen, button_color, button_rect)
    text = font.render(button_text, True, (text_color))
    screen.blit(text, (button_rect.x + (BUTTON_WIDTH - text.get_width()) // 2,
                       button_rect.y + (BUTTON_HEIGHT - text.get_height()) // 2))
    """
    button("connect",(WIDTH-390),10,100,35,RED,SCREEN,GRAY,WHITE)
    button("256Khz",(WIDTH-280),10,80,35,SCREEN,SCREEN,GRAY,WHITE,lambda: set_freq_range(256))
    button("512Khz",(WIDTH-190),10,80,35,SCREEN,SCREEN,GRAY,WHITE,lambda: set_freq_range(512))
    button("1024Khz",(WIDTH-100),10,80,35,SCREEN,SCREEN,GRAY,WHITE,lambda: set_freq_range(1024))

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
                        button_color = (RED)  
                        button_text = "Connect"
                        await toggle_connection()
                    else:
                        button_color = (GREEN)  
                        button_text = "Disconnect"
                        await toggle_connection()

        screen.fill((0, 0, 0))
        
        if is_connected:
            try:
                iq_data = await data_queue.get()
                complex_signal = process_signal(process_iq_data(iq_data), FREQ_RANGE)
                freqs, magnitudes = await compute_fft(complex_signal, SAMPLE_RATE)
                scaled_magnitudes = np.interp(
                    magnitudes, [MIN_DB, MAX_DB], [0, 255]
                ).astype(np.uint8)
                update_waterfall(waterfall_buffer, scaled_magnitudes)
                draw_waterfall(waterfall_surface, waterfall_buffer)
                draw_fft_graph(screen, freqs, scaled_magnitudes)
                #print(f"Magnitudes range: min={magnitudes.min()}, max={magnitudes.max()}")
                #print(f"Queue size: {data_queue.qsize()}/{data_queue.maxsize}")
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

