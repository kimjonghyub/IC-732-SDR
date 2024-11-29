import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np

# Bandwidth options
BANDWIDTH_OPTIONS = [200e3, 500e3, 960e3, 1.5e6]
current_bandwidth_idx = 0  # Index to track current bandwidth
current_bandwidth = BANDWIDTH_OPTIONS[current_bandwidth_idx]

# 데이터 생성 (샘플 데이터를 밴드폭에 따라 변화하도록 설정)
x = np.linspace(0, 10, 500)
y = np.sin(2 * np.pi * current_bandwidth * x / 1e6)

# 플롯 생성
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)  # 버튼 공간 확보
line, = ax.plot(x, y, label=f"Bandwidth: {current_bandwidth / 1e3} kHz")
ax.legend()

# 버튼 클릭 이벤트 함수
def update_bandwidth(event):
    global current_bandwidth_idx, current_bandwidth
    # 밴드폭 변경
    current_bandwidth_idx = (current_bandwidth_idx + 1) % len(BANDWIDTH_OPTIONS)
    current_bandwidth = BANDWIDTH_OPTIONS[current_bandwidth_idx]
    
    # 데이터 업데이트
    y_new = np.sin(2 * np.pi * current_bandwidth * x / 1e6)
    line.set_ydata(y_new)
    
    # 레이블 및 범위 업데이트
    ax.legend([f"Bandwidth: {current_bandwidth / 1e3} kHz"])
    ax.relim()
    ax.autoscale_view()
    plt.draw()

# 버튼 위치 및 속성 설정
button_ax = plt.axes([0.7, 0.05, 0.2, 0.075])  # [left, bottom, width, height]
bandwidth_button = Button(button_ax, "Change Bandwidth", color="lightgrey", hovercolor="grey")

# 버튼 클릭 시 실행할 함수 연결
bandwidth_button.on_clicked(update_bandwidth)

# 플롯 표시
plt.show()
