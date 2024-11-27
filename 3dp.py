import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.mplot3d import Axes3D
from scipy.fft import fft
from scipy.ndimage import gaussian_filter1d

CHUNK = 1024
RATE = 48000
min_freq = 400
max_freq = 4000
min_amp = 0.1     
max_amp = 7 

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=RATE,
                input=True,
                input_device_index=2, 
                frames_per_buffer=CHUNK)

dpi = 100
fig = plt.figure(figsize=(1024 / dpi, 600 / dpi), dpi=dpi)
fig.canvas.manager.set_window_title("3D Spectrum Waterfall")  

ax_3d = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

time_points = 45
x = np.arange(time_points)

freqs = np.fft.fftfreq(CHUNK, 1.0 / RATE) [:CHUNK // 2]
freq_indices = np.where((freqs >= min_freq) & (freqs <= max_freq))
filtered_freqs = freqs[freq_indices] / 1000

Z_data = np.zeros((len(filtered_freqs), time_points))
ax_3d.set_box_aspect([1, 1, 0.5])
ax_3d.view_init(elev=15, azim=0)


ax_3d.set_zlim(min_amp * 1.e6, max_amp * 1.e6)
ax_3d.zaxis.set_major_formatter(ScalarFormatter(useOffset=False))
ax_3d.zaxis.get_major_formatter().set_scientific(False)
ax_3d.set_xticks([])
ax_3d.set_zticks([])
ax_3d.set_ylabel('Frequency (kHz)', color="white") 
ax_3d.set_yticks(np.linspace(min_freq / 1000, max_freq / 1000, 6)) 
ax_3d.yaxis.labelpad = 15
ax_3d.tick_params(axis='x', colors='white')   
ax_3d.tick_params(axis='y', colors='white') 
ax_3d.tick_params(axis='z', colors='white')  


ax_3d.xaxis.pane.set_visible(False)
ax_3d.yaxis.pane.set_visible(False)
ax_3d.zaxis.pane.set_visible(False)
ax_3d.xaxis._axinfo['grid'].update({'linewidth': 0})
ax_3d.yaxis._axinfo['grid'].update({'linewidth': 0})
ax_3d.zaxis._axinfo['grid'].update({'linewidth': 0})

ax_3d.set_facecolor('black')
ax_3d.xaxis._axinfo["axisline"]["color"] = (1, 1, 1, 1)
ax_3d.yaxis._axinfo["axisline"]["color"] = (1, 1, 1, 1)
ax_3d.zaxis._axinfo["axisline"]["color"] = (1, 1, 1, 1)


X, Y = np.meshgrid(x, filtered_freqs)
surface = ax_3d.plot_surface(X, Y, Z_data, cmap='plasma')

try:
    while plt.fignum_exists(fig.number): 
        data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
        fft_data = np.abs(fft(data)[:CHUNK // 2])
        Z = gaussian_filter1d(fft_data[freq_indices], sigma=1.5)
        Z_data = np.roll(Z_data, -1, axis=1)
        Z_data[:, -1] = Z

        
        surface.remove()
        surface = ax_3d.plot_surface(X, Y, Z_data, cmap='plasma', rstride=1, cstride=1, antialiased=False)
        
        plt.pause(0.01)

except Exception as e:
    print(f"Error: {e}")

finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
    plt.close(fig)
    print("Audio stream and plot closed.")
