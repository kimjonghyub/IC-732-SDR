import asyncio
import struct
import pygame
import numpy as np
import scipy.signal
from scipy.interpolate import make_interp_spline
from scipy.signal import savgol_filter
import pygame.gfxdraw

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
CMD_RIG_SET_FREQ = 0x20
CMD_RIG_GET_FREQ = 0x21
FRAME_SIZE = 1024
BUFFER_SIZE = FRAME_SIZE * 128
FREQ_RANGE = 256

SERVER_IP = "222.117.38.98"
SERVER_PORT = 1234
RIG_PORT = 5678
FREQ = 69011500
SAMPLE_RATE = 1024000
GAIN = 496
FREQ_CORRECTION = 0
AGC_MODE = 1
DIRECT_SAMPLING = 0
GAIN_MODE = 0

WIDTH, HEIGHT = 400, 800
BUTTON_WIDTH, BUTTON_HEIGHT = 100, 35
FPS = 20
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
rig_command_queue = asyncio.Queue()
rig_response_queue = asyncio.Queue()

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
rig_connected = False
is_connected = False
reader = None
writer = None
rig_reader = None
rig_writer = None
tasks = []  

class SmoothValue:
    def __init__(self, smoothing_factor=0.1):
        self.smoothing_factor = smoothing_factor
        self.value = None

    def update(self, new_value):
        if self.value is None:
              self.value = new_value
        else:
              self.value = (self.smoothing_factor * new_value) + ((1 - self.smoothing_factor) * self.value)
        return self.value
center_magnitude_smoother = SmoothValue(smoothing_factor=0.1)
center_magnitude = 0
smoothed_center_magnitude = 0
def ctlbox():
    global smoothed_center_magnitude
    ctl_box = pygame.Surface(((WIDTH), BUTTON_AREA_HEIGHT) )
    ctl_box.fill(GRAY)
    screen.blit(ctl_box, (0,0))
    msg = "Server IP : %s" % (SERVER_IP) 
    screen.blit(font.render(msg, 1, WHITE, GRAY),(10,60))
    msg = "Port : %s" % (SERVER_PORT) 
    screen.blit(font.render(msg, 1, WHITE, GRAY),(220,60))
    msg = "Signal : %0.2f db" % (smoothed_center_magnitude) 
    screen.blit(font.render(msg, 1, WHITE, GRAY),(10,90))
    meter_rect = pygame.Rect(10, 120, 300, 20)  
    draw_level_meter(screen, smoothed_center_magnitude, meter_rect)
    
def draw_level_meter(screen, center_magnitude, meter_rect):
    min_dB = 0  
    max_dB = 150 
    normalized_value = (smoothed_center_magnitude - min_dB) / (max_dB - min_dB)
    normalized_value = np.clip(normalized_value, 0, 1)  
    pygame.draw.rect(screen, GRAY, meter_rect)
    fill_width = int(normalized_value * meter_rect.width)
    filled_rect = pygame.Rect(meter_rect.x, meter_rect.y, fill_width, meter_rect.height)
    pygame.draw.rect(screen, GREEN, filled_rect)
    pygame.draw.rect(screen, WHITE, meter_rect, 2)
    """
    font = pygame.font.SysFont(None, 24)
    magnitude_label = font.render(f"{center_magnitude:.2f} dB", True, WHITE)
    screen.blit(
        magnitude_label,
        (meter_rect.x + meter_rect.width + 10, meter_rect.y + (meter_rect.height - magnitude_label.get_height()) // 2)
    )
    """


button_rect = pygame.Rect((10, 10), (BUTTON_WIDTH, BUTTON_HEIGHT))
button_color = (RED) 
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
    
def remove_dc_component(iq_signal):
    real_part = np.real(iq_signal)
    imag_part = np.imag(iq_signal)

    real_part -= np.mean(real_part)
    imag_part -= np.mean(imag_part)

    return real_part + 1j * imag_part

def process_iq_data(iq_data):
    iq_data = np.frombuffer(iq_data, dtype=np.uint8).astype(np.float32)
    iq_data = (iq_data - 127.5) / 127.5
    real = iq_data[1::2]
    imag = iq_data[0::2]
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

def smooth_fft_graph(freqs, magnitudes, num_points=500):
    x_smooth = np.linspace(freqs.min(), freqs.max(), num_points)
    spline = make_interp_spline(freqs, magnitudes, k=3) 
    y_smooth = spline(x_smooth)
    return x_smooth, y_smooth

def smooth_magnitudes(magnitudes, window_length=11, polyorder=2):
    return savgol_filter(magnitudes, window_length, polyorder)
    
def draw_antialiased_line(screen, points, color):
    for i in range(len(points) - 1):
        pygame.draw.aaline(screen, color, points[i], points[i + 1])

def draw_fft_graph(screen, freqs, magnitudes):
    global center_magnitude, smoothed_center_magnitude
    x_smooth, y_smooth = smooth_fft_graph(freqs, magnitudes)
    y_smooth = smooth_magnitudes(y_smooth)
    graph_width = WIDTH - MARGIN
    graph_height = FFT_GRAPH_HEIGHT - MARGIN
    graph_x = MARGIN/2
    graph_y = BUTTON_AREA_HEIGHT + MARGIN/2
    pygame.draw.rect(screen, GRAY, (graph_x, graph_y, graph_width, graph_height))  
    mid_x = graph_x + graph_width // 2
    """
    x_coords = mid_x + (freqs / (SAMPLE_RATE / 2)) * (graph_width // 2)
    y_coords = graph_y + (graph_height * (1 - (magnitudes) / (graph_height)))
    y_coords = np.clip(y_coords, graph_y, graph_y + graph_height)
    """
    x_coords = graph_x + (x_smooth - freqs.min()) / (freqs.max() - freqs.min()) * graph_width
    """
    y_coords = graph_y + graph_height - (
        (y_smooth - magnitudes.min()+5) / (magnitudes.max() - magnitudes.min()) * graph_height
    )
    """
    denominator = magnitudes.max() - magnitudes.min()
    y_coords = graph_y + graph_height - (
    np.divide(
        (y_smooth - magnitudes.min() + 5),
        denominator,
        out=np.zeros_like(y_smooth),
        where=denominator != 0
    ) * graph_height
)
    y_coords = np.clip(y_coords, graph_y, graph_y + graph_height)
    points = np.column_stack((x_coords, y_coords)).astype(int)
    polygon_points = [(graph_x, graph_y + graph_height)] + points.tolist() + [
        (graph_x + graph_width, graph_y + graph_height)
    ]
    pygame.draw.polygon(screen, BLUE_GRAY, polygon_points)
    draw_antialiased_line(screen, points.tolist(), WHITE)
    
    # Draw x-axis
    pygame.draw.line(
        screen, RED, 
        (graph_x, graph_y + graph_height - 1), 
        (graph_x + graph_width, graph_y + graph_height - 1), 1)
    max_freq = SAMPLE_RATE / 1 / (1024 // FREQ_RANGE) 
    num_xticks = 4  
    tick_spacing = graph_width // num_xticks 
    
    for i in range(num_xticks + 1):
        x_pos = i * tick_spacing
        freq = int((i - num_xticks // 2) * (max_freq / num_xticks) / 1e3)  
        tick_label = f"{freq}kHz"
        pygame.draw.line(
            screen, DARK_RED, 
            (x_pos+graph_x, graph_y), #+ graph_height - 5), 
            (x_pos+graph_x, graph_y + graph_height + 5), 1)
        label = font_tick.render(tick_label, True, WHITE)
        label_rect = label.get_rect(center=(x_pos+graph_x, graph_y + graph_height + 15))
        screen.blit(label, label_rect)
        
    # Draw y-axis 
    num_yticks = 5
    tick_spacing = graph_height / (num_yticks - 1)

    for i in range(num_yticks):
        y_tick = graph_y + i * tick_spacing
        tick_value = magnitudes.min() + (magnitudes.max() - magnitudes.min()) * (num_yticks - 1 - i) / (num_yticks - 1)
        pygame.draw.line(
            screen, DARK_RED,
            (graph_x - 10, int(y_tick)),
            (graph_x+graph_width, int(y_tick)), 1)
        label = font_tick.render(f"{tick_value:.0f} dB", True, WHITE)
        screen.blit(label, (graph_x - 20, int(y_tick) - label.get_height() // 2))
    center_x = graph_x + (0 - freqs.min()) / (freqs.max() - freqs.min()) * graph_width
    center_index = np.argmin(np.abs(freqs - 0))
    raw_center_magnitude = magnitudes[center_index]
    smoothed_center_magnitude = center_magnitude_smoother.update(raw_center_magnitude)

async def set_freq_range(freq_range):
    global FREQ_RANGE, tasks
    FREQ_RANGE = freq_range
    if writer:
        try:
            #await send_command(writer, CMD_SET_SAMPLERATE, SAMPLE_RATE)
            #await send_command(writer, CMD_SET_FREQ_CORRECTION, FREQ_CORRECTION)
            await send_command(writer, CMD_SET_FREQ, FREQ)
            #await send_command(writer, CMD_SET_AGC_MODE, AGC_MODE)
            #await send_command(writer, CMD_SET_DIRECT_SAMPLING, DIRECT_SAMPLING)
            #await send_command(writer, CMD_SET_GAIN_MODE, GAIN_MODE)
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
    global reader, writer, rig_reader, rig_writer, is_connected, data_queue

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
        await close_rig_connection()  
    else:
        print("Connecting...")
        await handle_connection()
        await handle_rig_connection()  
        if is_connected:
            tasks.append(asyncio.create_task(receive_data(reader, data_queue)))

        
async def handle_rig_connection():
    global rig_reader, rig_writer, rig_connected
    try:
        rig_reader, rig_writer = await asyncio.open_connection(SERVER_IP, RIG_PORT)
        rig_connected = True
        print("Connected to RIG server.")
        asyncio.create_task(rig_reader_task())
    except Exception as e:
        print(f"Failed to connect to RIG server: {e}")
        rig_connected = False
        
async def rig_reader_task():
    global rig_reader, rig_writer, rig_connected
    try:
        while rig_connected:
            command, response_event = await rig_command_queue.get()
            try:
                rig_writer.write(command)
                await rig_writer.drain()
                response = await rig_reader.readexactly(4)  
                await rig_response_queue.put(response)
                response_event.set()  
            except Exception as e:
                print(f"Error while processing RIG command: {e}")
                response_event.set()  
    except asyncio.CancelledError:
        print("RIG reader task cancelled.")
    finally:
        rig_connected = False
        
async def send_rig_command(cmd_id, param):
    global rig_connected
    if not rig_connected:
        print("RIG server is not connected.")
        return None

    command = struct.pack(">BI", cmd_id, param)
    response_event = asyncio.Event()
    await rig_command_queue.put((command, response_event))
    await response_event.wait()  
    if not rig_response_queue.empty():
        response = await rig_response_queue.get()
        return response
    return None
        
async def close_rig_connection():
    global rig_reader, rig_writer, rig_connected, rig_command_queue
    if rig_writer:
        try:
            rig_writer.close()
            await rig_writer.wait_closed()
        except Exception as e:
            print(f"Error closing RIG server connection: {e}")
    rig_reader, rig_writer = None, None
    rig_connected = False
    while not rig_command_queue.empty():
        try:
            rig_command_queue.get_nowait()
        except asyncio.QueueEmpty:
            break
    
async def toggle_rig_connection():
    global rig_connected
    if rig_connected:
        print("Disconnecting from RIG server...")
        await close_rig_connection()
    else:
        print("Connecting to RIG server...")
        await handle_rig_connection()
        
async def get_rig_frequency():
    response = await send_rig_command(CMD_RIG_GET_FREQ, 0)
    if response is not None:
        try:
            rig_frequency = struct.unpack(">I", response)[0]
            if rig_frequency == 0:
                print("RIG server returned an error.")
            elif 1_000_000 <= rig_frequency <= 3_000_000_000:
                print(f"Current rig frequency: {rig_frequency} Hz")
            else:
                print(f"Invalid frequency received: {rig_frequency} Hz")
        except struct.error as e:
            print(f"Failed to decode frequency data: {e}")
    else:
        print("Failed to get a response from the RIG server.")
    
        
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
    button("Rig Freq", (WIDTH - 390), 60, 80, 35, BLUE_GRAY, SCREEN, WHITE, RED, get_rig_frequency)
    

        
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
        ctlbox()
        if is_connected:
            try:
                iq_data = await data_queue.get()
                complex_signal = process_signal(process_iq_data(iq_data), FREQ_RANGE)
                complex_signal = remove_dc_component(complex_signal)
                freqs, magnitudes = await compute_fft(complex_signal, SAMPLE_RATE)
                magnitudes = np.clip(magnitudes, -30, 100)
                exponent = 4  
                processed_magnitudes = (magnitudes - magnitudes.min()) ** exponent
                scaled_magnitudes = np.interp(
                    processed_magnitudes, [processed_magnitudes.min(), processed_magnitudes.max()], [0, 255]
                ).astype(np.uint8)
                #scaled_magnitudes = np.interp(
                #    magnitudes, [MIN_DB, MAX_DB], [0, 255]
                #).astype(np.uint8)
                wf_magnitudes = np.interp(magnitudes, [MIN_DB, MAX_DB], [0, 255]).astype(np.uint8)
                update_waterfall(waterfall_buffer, wf_magnitudes)
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

