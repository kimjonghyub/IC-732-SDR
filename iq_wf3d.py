import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def palette_color(val, vmin, vmax):
    """Translate a data value into a color based on the intensity."""
    f = (float(val) - vmin) / (vmax - vmin)     # between 0 and 1
    r = int(min(255, max(0, f * 255)))
    g = int(min(255, max(0, (1 - f) * 255)))
    b = 128  # Static for simplicity
    return (r / 255, g / 255, b / 255)  # Normalize to [0, 1] for matplotlib

class Waterfall3D:
    def __init__(self, freq_bins, time_steps, vmin, vmax):
        """Initialize the 3D waterfall display."""
        self.freq_bins = freq_bins
        self.time_steps = time_steps
        self.vmin = vmin
        self.vmax = vmax
        self.data = np.zeros((time_steps, freq_bins))
        self.time_index = 0

    def update(self, spectrum):
        """Update the waterfall with new spectrum data."""
        self.data[self.time_index % self.time_steps] = spectrum
        self.time_index += 1

    def display(self):
        """Render the 3D waterfall plot."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x = np.arange(self.freq_bins)
        y = np.arange(self.time_steps)
        X, Y = np.meshgrid(x, y)
        Z = self.data

        # Normalize color for the Z intensity levels
        colors = np.array([palette_color(val, self.vmin, self.vmax) for row in Z for val in row])
        colors = colors.reshape(Z.shape + (3,))

        # Plot the surface
        ax.plot_surface(X, Y, Z, facecolors=colors, rstride=1, cstride=1, linewidth=0, antialiased=False)
        ax.set_xlabel('Frequency')
        ax.set_ylabel('Time')
        ax.set_zlabel('Intensity')
        plt.show()

# Sample usage with synthetic data
freq_bins = 100
time_steps = 50
vmin, vmax = -20, 20
waterfall = Waterfall3D(freq_bins, time_steps, vmin, vmax)

# Generate random data for demonstration
for _ in range(time_steps):
    spectrum = np.random.uniform(vmin, vmax, freq_bins)
    waterfall.update(spectrum)

waterfall.display()
