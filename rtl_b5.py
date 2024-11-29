import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.mplot3d import Axes3D
from rtlsdr import RtlSdr
from scipy.fft import fft
from scipy.ndimage import gaussian_filter1d

CENTER_FREQ = 69.011500e6 
SAMPLE_RATE = 2.56e6 
GAIN = 49.6  
NFFT = 1024
CHUNK = 1024 
min_amp = -25  
max_amp = 25
bandwidth = 960e3


sdr = RtlSdr()
sdr.center_freq = CENTER_FREQ
sdr.sample_rate = SAMPLE_RATE
sdr.gain = GAIN

dpi = 100
fig = plt.figure(figsize=(1024 / dpi, 600 / dpi), dpi=dpi)
fig.canvas.manager.set_window_title("3D Spectrum Waterfall")



ax_3d = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

time_points = 25
x = np.arange(time_points)

freqs = np.fft.fftshift(np.fft.fftfreq(NFFT, 1 / SAMPLE_RATE))
freq_indices = (freqs >= -bandwidth / 2) & (freqs <= bandwidth / 2)
filtered_freqs = freqs[freq_indices] / 1e3

Z_data = np.zeros((len(filtered_freqs), time_points))
ax_3d.set_box_aspect([8, 8, 0.8])
#ax_3d.view_init(elev=15, azim=0)
ax_3d.view_init(elev=25, azim=25)

ax_3d.set_zlim(min_amp, max_amp)
ax_3d.zaxis.set_major_formatter(ScalarFormatter(useOffset=False))
ax_3d.zaxis.get_major_formatter().set_scientific(False)
ax_3d.set_xticks([])
ax_3d.set_zticks([])
ax_3d.set_ylabel('Frequency (kHz)', color="white")
ax_3d.set_ylim(filtered_freqs[0], filtered_freqs[-1])
ax_3d.yaxis.labelpad = 10
ax_3d.tick_params(axis='x', colors='white')
ax_3d.tick_params(axis='y', colors='white')
ax_3d.tick_params(axis='z', colors='white')

ax_3d.xaxis.pane.set_visible(False)
ax_3d.yaxis.pane.set_visible(False)
ax_3d.zaxis.pane.set_visible(False)
ax_3d.xaxis._axinfo['grid'].update({'linewidth': 1})
ax_3d.yaxis._axinfo['grid'].update({'linewidth': 1})
ax_3d.zaxis._axinfo['grid'].update({'linewidth': 1})

ax_3d.set_facecolor('black')
ax_3d.xaxis._axinfo["axisline"]["color"] = (1, 1, 1, 1)
ax_3d.yaxis._axinfo["axisline"]["color"] = (1, 1, 1, 1)
ax_3d.zaxis._axinfo["axisline"]["color"] = (1, 1, 1, 1)

X, Y = np.meshgrid(x, filtered_freqs)
surface = ax_3d.plot_surface(X, Y, Z_data, cmap='plasma')

try:
    while plt.fignum_exists(fig.number):
        samples = sdr.read_samples(CHUNK * 4)  
        fft_vals = np.fft.fft(samples, NFFT)  # Compute FFT
        fft_vals = np.fft.fftshift(fft_vals)  # Center FFT around 0
        fft_data = 10 * np.log10(np.abs(fft_vals) ** 2 + 1e-10)  # Convert to dB scale
        Z = gaussian_filter1d(fft_data[freq_indices], sigma=1.5)
        Z_data = np.roll(Z_data, -1, axis=1)
        Z_data[:, -1] = Z
        
        
        surface.remove()
        surface = ax_3d.plot_surface(X, Y, Z_data, cmap='plasma', rstride=1, cstride=1, antialiased=False)
        
        plt.pause(0.01)

except Exception as e:
    print(f"Error: {e}")

finally:
    sdr.close()
    plt.close(fig)
    print("SDR and plot closed.")
