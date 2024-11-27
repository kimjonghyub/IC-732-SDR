import pygame
import pyaudio
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import numpy as np
from scipy.fft import fft
from scipy.ndimage import gaussian_filter

# Precomputed LUT for colors (256 steps)
color_lut = np.array([[z, 1 - z, z * 0.5] for z in np.linspace(0, 1, 256)])

def calculate_color(z_value):
    
    red = z_value 
    green = 0
    blue = 1 - z_value  
    return red, green, blue 
    """
    index = int(np.clip(z_value * 255, 0, 255))  # Map 0-1 range to 0-255
    return color_lut[index]
    """


# Constants
CHUNK = 1024
RATE = 44100

# PyAudio setup
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=RATE,
                input=True,
                input_device_index=2, 
                frames_per_buffer=CHUNK)

# Pygame and OpenGL setup
pygame.init()
glutInit()
display = (1024, 600)
pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
gluPerspective(45, (display[0] / display[1]), 0.01, 10)
glTranslatef(-2.75, -1.2, -3.4)
glRotatef(-90, 1, 0, 0)
glRotatef(0, 0, 0, 1)

# Data dimensions
grid_size = 50
min_freq = 500  
max_freq = 5000
freqs = np.fft.fftfreq(CHUNK, 1.0 / RATE)
freq_indices = np.where((freqs >= min_freq) & (freqs <= max_freq))[0]
selected_freqs = freqs[freq_indices] / 1000
X, Y = np.meshgrid(selected_freqs, np.linspace(5, 0, grid_size))
Z = np.zeros_like(X)
current_line = 0  

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
"""
def draw_surface(X, Y, Z):
    
    for i in range(Z.shape[0]):
        glBegin(GL_LINE_STRIP)
        for j in range(Z.shape[1]):
            color = calculate_color(Z[i][j])
            glColor3fv(color)
            glVertex3fv((X[i][j], Y[i][j], Z[i][j]))
        glEnd()
"""

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
    bar_width = 0.01  # Bar width
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
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == KEYDOWN and event.key == K_q:
                running = False

        # Audio input and FFT processing
        data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
        fft_data = np.abs(fft(data))[:CHUNK // 2]
        selected_fft_data = fft_data[freq_indices]
        normalized_fft = selected_fft_data / np.max(selected_fft_data)

        # Update data
        X, Y, Z = update_data(X, Y, Z, normalized_fft, step=0.1)
        x_left = np.min(selected_freqs) - 0.1  
        x_right = np.max(selected_freqs) + 0.1  
        # Render scene
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        draw_surface(X, Y, Z, x_left=x_left, x_right=x_right, z_wall_fixed=1)

        # Draw axis and labels
        draw_text_opengl((-3, 5, 1.5), "Frequency Spectrum", color=(1, 1, 1))
        for i, freq in enumerate(np.linspace(min_freq, max_freq, 6) / 1000):  
            draw_text_opengl((i * 1.03, 0.6, -0.4), f"{freq:.1f} kHz", color=(0.7, 0.7, 0.7))

        pygame.display.flip()
        pygame.time.wait(10)

except Exception as e:
    print(f"Error: {e}")

finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
    pygame.quit()
    print("Audio stream and plot closed.")
