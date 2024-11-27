import pygame as pg
import pyaudio
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import numpy as np
from scipy.fft import fft
from scipy.ndimage import gaussian_filter

# Pygame and OpenGL setup
pg.init()
pg.font.init()
glutInit()
display = (1024, 600)
surf_main = pg.display.set_mode(display, DOUBLEBUF | OPENGL)
#surf_main = pg.display.set_mode((1024,600))
pg.display.set_caption("IC732-SDR")
gluPerspective(45, (display[0] / display[1]), 0.01, 10)
glTranslatef(-2.75, -1.2, -3.4)
glRotatef(-90, 1, 0, 0)
glRotatef(0, 0, 0, 1)

# Constants
CHUNK = 1024
RATE = 48000

# Data dimensions
grid_size = 83
min_freq = 500  
max_freq = 5000
freqs = np.fft.fftfreq(CHUNK, 1.0 / RATE)
freq_indices = np.where((freqs >= min_freq) & (freqs <= max_freq))[0]
selected_freqs = freqs[freq_indices] / 1000
X, Y = np.meshgrid(selected_freqs, np.linspace(5, 0, grid_size))
Z = np.zeros_like(X)
current_line = 0  

# Precomputed LUT for colors (256 steps)
color_lut = np.array([[z, 1 - z, z * 0.5] for z in np.linspace(0, 1, 256)])
"""
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
"""

BLACK =    (0.0, 0.0, 0.0)
WHITE =    (1.0, 1.0, 1.0)
GREEN =    (0.0, 1.0, 0.0)
BLUE =     (0.0, 0.0, 1.0)
RED =      (1.0, 0.0, 0.0)
YELLOW =   (0.75, 0.75, 0.0)
DARK_RED = (0.5, 0.0, 0.0)
LITE_RED = (1.0, 0.39, 0.39)
BGCOLOR =  (1.0, 0.9, 0.78)
BLUE_GRAY = (0.39, 0.39, 0.71)
ORANGE =   (1.0, 0.59, 0.0)
GRAY =     (0.24, 0.24, 0.24)
SCREEN =   (0.996, 0.647, 0.0)

lgfont = pg.font.Font("Patopian 1986.ttf", 100) # normal 16
lgfont_ht = lgfont.get_linesize()       # text height
mefont = pg.font.Font("BITSUMIS.TTF", 25)
mefont_ht = mefont.get_linesize()
mzfont = pg.font.Font("BITSUMIS.TTF", 30)
mzfont_ht = mzfont.get_linesize()


def calculate_color(z_value):
    
    red = z_value 
    green = 0
    blue = 1 - z_value  
    return red, green, blue 
    """
    index = int(np.clip(z_value * 255, 0, 255))  # Map 0-1 range to 0-255
    return color_lut[index]
    """




# PyAudio setup
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=RATE,
                input=True,
                input_device_index=2, 
                frames_per_buffer=CHUNK)

def get_mic_data():
    global audio_data,normalized_fft,level
    try:
        # Audio input and FFT processing
        audio_data = stream.read(CHUNK, exception_on_overflow=False)
        data = np.frombuffer(audio_data, dtype=np.int16)
        fft_data = np.abs(fft(data))[:CHUNK // 2]
        max_amplitude = np.max(np.abs(data))
        selected_fft_data = fft_data[freq_indices]
        normalized_fft = selected_fft_data / np.max(selected_fft_data)
        level = np.clip(np.abs(data).mean() / 32768, 0, 1)
        
        if max_amplitude > 30000:  
            data = np.clip(data, -30000, 30000)
        return data

    except IOError as e:
        print(f"Audio buffer error: {e}")
        return np.zeros(CHUNK, dtype=np.int16)  

    except Exception as e:
        print(f"Unexpected error in get_mic_data_safe: {e}")
        return np.zeros(CHUNK, dtype=np.int16)

def setup_2d_viewport():
    """
    Configure the OpenGL viewport for 2D rendering on the top portion of the screen.
    """
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()

    # Set orthographic projection for the top 200 pixels
    glOrtho(0, display[0], 0, display[1], -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()


def restore_3d_viewport():
    """
    Restore the 3D viewport and perspective projection.
    """
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)
    glPopMatrix()


def draw_2d_overlay():
    """
    Render 2D content in the overlay region.
    """
    
    glColor3f(1.0,0.59,0.0)
    glBegin(GL_QUADS)

    # Example: Draw a white rectangle as a background for the 2D area
    glVertex2f(0, display[1])            
    glVertex2f(display[0], display[1])    
    glVertex2f(display[0], display[1] - 200)  
    glVertex2f(0, display[1] - 200)           
    glEnd()
    
    draw_text(
        pg.display.get_surface(),
        "MODE LSB / BAND AM BORADCAST",
        position=(10,display[1]-35),
        font = mefont,
        color=(0,0,0),
        )
    draw_text(
        pg.display.get_surface(),
        "VFO B",
        position=(10,display[1]-60),
        font = mefont,
        color=(0,0,0),
        )
        
    draw_text(
        pg.display.get_surface(),
        #unicode_data, 
        "001.135.00",
        position=(240,display[1]-170),
        font = lgfont,
        color=(0,0,0),
        )   
        
    draw_text(
        pg.display.get_surface(),
        "MHZ",
        position=(800,display[1]-160),
        font = mzfont,
        color=(0,0,0),
        )   
        
    draw_text(
        pg.display.get_surface(),
        "STEP  BAND  MODE  VFO A/B  QUIT",
        position=(530,display[1]-190),
        font = mefont,
        color=(0,0,0),
        )
    #msg = "Mhz"
    #surf_main.blit(mzfont.render(msg, 1, BLACK, SCREEN), (220, 100))
    # Draw some 2D text or other 2D elements
    #draw_text_2d((10, display[1] - 30), "MODE AM / BAND BROADCAST", (0, 0, 0))
    #draw_text_2d((10, display[1] - 70), "VFO A", (0, 0, 0))
    #draw_text_2d((220, display[1] - 90), "001.135.00 mHz", (0, 0, 0))
    #draw_text_2d((600, display[1] - 180), "STEP  BAND  MODE  VFO A/B  QUIT", (0, 0, 0))
    
def draw_2d_overlay_with_graph(graph_data):
    """
    Render 2D content in the overlay region with a line graph.
    """
    # Draw the 2D background area
    glColor3f(0.1, 0.1, 0.1)  # White background
    glBegin(GL_QUADS)
    glVertex2f(0, display[1] - 300)
    glVertex2f(display[0], display[1] - 300)
    glVertex2f(display[0], display[1] - 200)
    glVertex2f(0, display[1] - 200)
    glEnd()

    # Draw graph axes
    glColor3f(0, 0, 0)  # Black axes
    glLineWidth(2.0)
    glBegin(GL_LINES)
    # X-axis
    glVertex2f(20, display[1] - 290)
    glVertex2f(display[0] - 20, display[1] - 290)
    # Y-axis
    glVertex2f(20, display[1] - 290)
    glVertex2f(20, display[1] - 220)
    glEnd()
    """
    # Draw the line graph
    glColor3f(0, 0, 1)  # Blue line
    glLineWidth(1)
    glBegin(GL_LINE_STRIP)
    for i, value in enumerate(graph_data):
        x = 20 + i * ((display[0] - 30) / len(graph_data))  # Scale X values
        y = (display[1] - 290) + value * 70  # Scale Y values (adjust multiplier for scaling)
        glVertex2f(x, y)
    glEnd()
    """
    # Draw the bar graph
    bar_width = ((display[0] - 40) / len(graph_data)) - 2  # Dynamic bar width
    for i, value in enumerate(graph_data):
        x_start = 20 + i * ((display[0] - 40) / len(graph_data))  # Start of bar
        x_end = x_start + bar_width  # End of bar
        y = (display[1] - 290) + value * 70  # Height of bar (adjust multiplier for scaling)

        # Draw bar as a filled rectangle
        glColor3f(0.5, 0.5, 0.5)  # Blue color for bars
        glBegin(GL_QUADS)
        glVertex2f(x_start, display[1] - 290)  # Bottom-left
        glVertex2f(x_end, display[1] - 290)    # Bottom-right
        glVertex2f(x_end, y)                   # Top-right
        glVertex2f(x_start, y)                 # Top-left
        glEnd()

    # Draw text on the overlay
    #draw_text_2d((10, display[1] - 260), "2D Test Text", (0, 0, 0))  # Example text
    
    


def draw_text_2d(position, text, color=(1, 1, 1)):
    """
    Render text in 2D overlay space using OpenGL.
    """
    glColor3f(*color)
    glRasterPos2f(*position)
    for char in text:
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(char))

def draw_text(surface, text, position, font, color=(1,1,1)):
    #font = pg.font.Font(None, font_size)
    text_surface = font.render(text, True, color)
    text_data = pg.image.tostring(text_surface, "RGBA", True)
    x, y = position
    glRasterPos2f(x, y)
    glDrawPixels(
        text_surface.get_width(),
        text_surface.get_height(),
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        text_data,
    )
        
def update_data(X, Y, Z, new_fft, step=0.1, sigma=1.0):
    """
    Update Y and Z with smoothed data.
    """
    global current_line
    Y += step  # Move Y values up
    if np.max(Y[current_line, :]) > 5:  # Reset line if it moves out of bounds
        Y[current_line, :] = -0.1
        Z[current_line, :] = gaussian_filter(new_fft, sigma=sigma)
        current_line = (current_line + 1) % Y.shape[0]
    return X, Y, Z


def draw_grid_walls(x_left, x_right, z_fixed=5, grid_spacing=0.5, height=10, line_width=2.0):
   
    # Enable line smoothing for smoother lines
    glEnable(GL_LINE_SMOOTH)
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glEnable(GL_BLEND)

    # Set line width
    glLineWidth(line_width)

    # Draw vertical lines
    for y in np.arange(0, height + grid_spacing, grid_spacing):
        brightness = 1 - y / height  
        glColor3f(brightness * 0.2, brightness * 0.2, brightness * 0.2)  

        
        glBegin(GL_LINES)
        glVertex3f(x_left, y, 0)
        glVertex3f(x_left, y, z_fixed)
        glEnd()

        
        glBegin(GL_LINES)
        glVertex3f(x_right, y, 0)
        glVertex3f(x_right, y, z_fixed)
        glEnd()

    # Draw horizontal lines
    for z in np.arange(0, z_fixed + grid_spacing, grid_spacing):
        glBegin(GL_LINES)
        for y in np.arange(0, height + grid_spacing, grid_spacing):
            brightness = 1 - y / height  
            glColor3f(brightness * 0.2, brightness * 0.2, brightness * 0.2)  

            
            glVertex3f(x_left, y, z)
            glVertex3f(x_left, y + grid_spacing, z)

            
            glVertex3f(x_right, y, z)
            glVertex3f(x_right, y + grid_spacing, z)
        glEnd()

    # Reset line width to default
    glLineWidth(1.0)
    glDisable(GL_LINE_SMOOTH)


    for z in np.arange(0, z_fixed + grid_spacing, grid_spacing):
        glBegin(GL_LINES)
        for y in np.arange(0, height + grid_spacing, grid_spacing):
            brightness = 0.8 - y / height  
            glColor3f(brightness * 0.5, brightness * 0.5, brightness * 0.5)  

            
            glVertex3f(x_left, y, z)
            glVertex3f(x_left, y + grid_spacing, z)

            
            glVertex3f(x_right, y, z)
            glVertex3f(x_right, y + grid_spacing, z)
        glEnd()

def draw_surface(X, Y, Z, x_left, x_right, z_wall_fixed=5):
    draw_grid_walls(x_left, x_right, z_fixed=z_wall_fixed, grid_spacing=0.2, height=10)
    bar_width = 0.02  # Bar width
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            # Calculate color for the bar
            color = calculate_color(Z[i][j])
            glColor3fv(color)

            # Define bar position and size
            x = X[i][j]
            y = Y[i][j]
            z = Z[i][j]
            
            # Render the bar as a 3D quad
            glBegin(GL_QUADS)
            
            # Bottom face
            glVertex3fv((x - bar_width / 2, y, 0))
            glVertex3fv((x + bar_width / 2, y, 0))
            
            # Top face
            glVertex3fv((x + bar_width / 2, y, z))
            glVertex3fv((x - bar_width / 2, y, z))
            
            glEnd()
            

def draw_text_opengl(position, text, color=(1, 1, 1)):
    """
    Draw 2D text using OpenGL.
    """
    glColor3f(*color)
    glRasterPos3f(*position)
    for char in text:
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(char))

# Main loop
running = True
try:
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    while running:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
            elif event.type == KEYDOWN and event.key == K_q:
                running = False

        get_mic_data()
        unicode_data = f"{level:.8f}"
        
        # Update data
        X, Y, Z = update_data(X, Y, Z, normalized_fft, step=0.1)
        x_left = np.min(selected_freqs) - 0.1
        x_right = np.max(selected_freqs) + 0.1

        # Render scene
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # 2D overlay rendering
        setup_2d_viewport()
        draw_2d_overlay()
        draw_2d_overlay_with_graph(normalized_fft)
        restore_3d_viewport()

        # 3D rendering
        draw_surface(X, Y, Z, x_left=x_left, x_right=x_right, z_wall_fixed=1)

        # Draw axis and labels
        #draw_text_opengl((-3, 5, 1.5), "Frequency Spectrum", color=(1, 1, 1))
        for i, freq in enumerate(np.linspace(min_freq, max_freq, 6) / 1000):
            draw_text_opengl((i * 1.03, 0.6, -0.4), f"{freq:.1f} kHz", color=(0.7, 0.7, 0.7))

        

        pg.display.flip()
        pg.time.wait(10)

except Exception as e:
    print(f"Error: {e}")

finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
    pg.quit()
    print("Audio stream and plot closed.")
