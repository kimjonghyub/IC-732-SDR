import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from rtlsdr import RtlSdr
from scipy.ndimage import gaussian_filter1d


CENTER_FREQ = 69.011500e6
SAMPLE_RATE = 2.56e6
GAIN = 49.6
NFFT = 2028
CHUNK = 1024
min_amp = -10
max_amp = 30
bandwidth = 960e3
time_points = 50  


sdr = RtlSdr()
sdr.center_freq = CENTER_FREQ
sdr.sample_rate = SAMPLE_RATE
sdr.gain = GAIN


freqs = np.fft.fftshift(np.fft.fftfreq(NFFT, 1 / SAMPLE_RATE))
freq_indices = np.where((freqs >= -bandwidth / 2) & (freqs <= bandwidth / 2))[0]
filtered_freqs = freqs[freq_indices] / 1e3  

Z_data = np.zeros((time_points, len(filtered_freqs)))


dpi = 100
fig = plt.figure(figsize=(12, 6), dpi=dpi)
fig.canvas.manager.set_window_title("3D Moving Frequency-Power Graph")

ax_3d = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  
ax_3d.set_xlim(filtered_freqs[0], filtered_freqs[-1])  
ax_3d.set_ylim(time_points - 1, 0)  
ax_3d.set_zlim(min_amp, max_amp) 
ax_3d.set_box_aspect([8, 4, 0.8])
ax_3d.view_init(elev=25, azim=100)

ax_3d.set_xlabel('Frequency (kHz)', color='white')
ax_3d.set_ylabel('Time (Steps)', color='white')
ax_3d.set_zlabel('Power (dB)', color='white')
ax_3d.tick_params(axis='x', colors='white')
ax_3d.tick_params(axis='y', colors='white')
ax_3d.tick_params(axis='z', colors='white')
ax_3d.set_facecolor('black')


lines = []
for t_idx in range(time_points):
    line, = ax_3d.plot([], [], [], lw=1, color='cyan', alpha=0.7)
    lines.append(line)

try:
    while plt.fignum_exists(fig.number):
       
        samples = sdr.read_samples(CHUNK * 4)
        fft_vals = np.fft.fft(samples, NFFT)
        fft_vals = np.fft.fftshift(fft_vals)
        fft_data = 10 * np.log10(np.abs(fft_vals) ** 2 + 1e-10)
        Z = gaussian_filter1d(fft_data[freq_indices], sigma=1.5)
        
        
        Z_data = np.roll(Z_data, 1, axis=0)
        Z_data[0, :] = Z

        
        for t_idx, line in enumerate(lines):
            line.set_data(filtered_freqs, [t_idx] * len(filtered_freqs))
            line.set_3d_properties(Z_data[t_idx, :])

       
        fig.canvas.draw_idle()
        plt.pause(0.01)

except Exception as e:
    print(f"Error: {e}")

finally:
    sdr.close()
    plt.close(fig)
    print("SDR and plot closed.")
