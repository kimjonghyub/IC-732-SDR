import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import numpy as np
import pyaudio
from scipy.fft import fft

CHUNK = 1024
RATE = 44100


p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=RATE,
                input=True,
                input_device_index=2, 
                frames_per_buffer=CHUNK)

pygame.init()
glutInit()
display = (1024, 600)
pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
gluPerspective(45, (display[0] / display[1]), 0.01, 20.0)
glTranslatef(-2.2, -1, -3)


glRotatef(-90, 1, 0, 0) 
glRotatef(0, 0, 0, 1) 
#glRotatef(-45, 0, 1, 0) 

grid_size = 83
min_freq = 400  
max_freq = 4000 
freqs = np.fft.fftfreq(CHUNK, 1.0 / RATE)
freq_indices = np.where((freqs >= min_freq) & (freqs <= max_freq))[0]
selected_freqs = freqs[freq_indices] / 1000

X, Y = np.meshgrid(selected_freqs, np.linspace(5, 0, len(selected_freqs)))
Z = np.zeros_like(X)

def draw_text_opengl(position, text, color=(1, 1, 1)):
    """Draw 2D text using OpenGL."""
    glColor3f(*color)
    glRasterPos3f(*position)
    for char in text:
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(char))



def draw_axes():
    
    glBegin(GL_LINES)
    glColor3fv((1, 0, 0)) #x red
    glVertex3fv((0, 0, 0))
    glVertex3fv((5, 0, 0))
    glColor3fv((0, 0, 0)) #y green
    glVertex3fv((0, 0, 0))
    glVertex3fv((0, 5, 0))
    glColor3fv((0, 0, 0)) #z blue
    glVertex3fv((0, 0, 0))
    glVertex3fv((0, 0, 5))
    glEnd()
    """
    glColor3fv((1, 1, 1))  
    for i, freq in enumerate(np.linspace(min_freq, max_freq, 5) / 1000):  
        draw_text((i * 1.0, 0, 0), f"{freq:.1f} kHz", font_size=14)
    """
"""
def draw_surface(Z):
        
    glBegin(GL_POINTS)
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            color = (0.5, 0.5 + Z[i][j] * 0.5, 0.5 - Z[i][j] * 0.5)
            glColor3fv(color)
            glVertex3fv((X[i][j], Y[i][j], Z[i][j]))
    glEnd()
"""
def draw_surface(Z):
    glBegin(GL_QUADS)
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1] - 1):  
            color = (0.5, 0.5 + Z[i][j] * 0.5, 0.5 - Z[i][j] * 0.5)
            glColor3fv(color)

            
            x1, y1, z1 = X[i][j], Y[i][j], 0
            x2, y2, z2 = X[i][j], Y[i][j], Z[i][j]
            x3, y3, z3 = X[i][j + 1], Y[i][j + 1], Z[i][j + 1]
            x4, y4, z4 = X[i][j + 1], Y[i][j + 1], 0

            glVertex3fv((x1, y1, z1))
            glVertex3fv((x2, y2, z2))
            glVertex3fv((x3, y3, z3))
            glVertex3fv((x4, y4, z4))
    glEnd()


running = True
try:
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == KEYDOWN and event.key == K_q:
                running = False
    
        data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
        fft_data = np.abs(fft(data))[:CHUNK // 2]
        
        selected_fft_data = fft_data[freq_indices]
        normalized_fft = selected_fft_data / np.max(selected_fft_data)
        Z[:, :] = np.tile(normalized_fft, (grid_size, 1))
    
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        draw_axes()
        draw_surface(Z)
        
        draw_text_opengl((-3, 5, 1.5), "Frequency Spectrum", color=(1, 1, 1))
        for i, freq in enumerate(np.linspace(min_freq, max_freq, 5) / 1000):  
            draw_text_opengl((i * 1.0, 0.38, -0.3), f"{freq:.1f} kHz", color=(1, 1, 1))
        
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


