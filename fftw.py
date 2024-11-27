import subprocess
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


rtl_command = [
    "rtl_power_fftw",
    "-f", "69011500",  
    "-n", "1024",          
    "-g", "496"             
]


fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2)
ax.set_xlim(88, 108)  
ax.set_ylim(-100, 0)  
ax.set_xlabel("Frequency (MHz)")
ax.set_ylabel("Power (dB)")
ax.set_title("Real-time RTL-SDR Spectrum")


frequencies = np.linspace(88, 108, 200)  
power_data = np.full_like(frequencies, -100)  


def update(frame):
    global power_data
    try:
        
        with subprocess.Popen(rtl_command, stdout=subprocess.PIPE, text=True) as proc:
            while True:
                line_data = proc.stdout.readline()  
                if not line_data:
                    break  
                
                
                if line_data.startswith('#') or line_data.strip() == "":
                    continue

               
                try:
                    values = np.array(list(map(float, line_data.split(' '))))
                    freqs = values[0::2] / 1e6  
                    power = values[1::2]        
                    power_data = np.interp(frequencies, freqs, power) 
                    
                    
                    line.set_data(frequencies, power_data)
                    break  
                except ValueError:
                    continue 
    except Exception as e:
        print(f"Error: {e}")

    return line,


ani = FuncAnimation(fig, update, interval=10, blit=True)

plt.show()
