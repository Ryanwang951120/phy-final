import numpy as np
import matplotlib
matplotlib.use('TkAgg') # 強制使用 TkAgg 後端以確保彈出視窗
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox
import matplotlib.animation as animation
import tkinter.messagebox as messagebox
import json
import os

# 設定初始參數
L = 10.0          # 弦長 (m)
init_n = 1        # 諧波數 (n)
init_T = 100.0    # 張力 (N)
init_rho = 1.0    # 線密度 (kg/m)
init_amp = 1.0    # 振幅 (m)

# 建立 x 軸數據
x = np.linspace(0, L, 1000)

# 建立圖表
fig, (ax, ax_rel) = plt.subplots(1, 2, figsize=(15, 8))
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.4, wspace=0.2) # 增加底部空間給更多拉桿

# 初始波形
line, = ax.plot(x, np.zeros_like(x), lw=2, color='blue', label='Standing Wave (y1+y2)')
line1, = ax.plot(x, np.zeros_like(x), lw=1, color='green', alpha=0.5, linestyle='--', label='Right Traveling (y1)')
line2, = ax.plot(x, np.zeros_like(x), lw=1, color='red', alpha=0.5, linestyle='--', label='Left Traveling (y2)')

ax.set_ylim(-4, 4)
ax.set_xlim(0, L)
ax.set_title('Standing Wave Simulation (Fixed Ends)')
ax.set_xlabel('Position x (m)')
ax.set_ylabel('Amplitude y (m)')
ax.legend(loc='upper right')
ax.grid(True)

# T-f 關係圖
T_vals = np.linspace(0, 500, 200)
# 初始曲線
# f = (n / (2L)) * sqrt(T / rho)
def get_f_curve(n, rho, T_array):
    return (n / (2 * L)) * np.sqrt(T_array / rho)

f_vals = get_f_curve(init_n, init_rho, T_vals)
line_curve, = ax_rel.plot(T_vals, f_vals, 'b-', label=r'$f \propto \sqrt{T}$')

# 當前點
current_v = np.sqrt(init_T / init_rho)
current_f = current_v / (2 * L / init_n)
point_curr, = ax_rel.plot([init_T], [current_f], 'ro', label='Current State')

ax_rel.set_title('Frequency vs Tension')
ax_rel.set_xlabel('Tension T (N)')
ax_rel.set_ylabel('Frequency f (Hz)')
ax_rel.set_xlim(0, 520)
ax_rel.legend()
ax_rel.grid(True)

# 建立拉桿軸
# 調整位置以容納更長的標籤，並避免與資訊框重疊
axcolor = 'lightgoldenrodyellow'
ax_n = plt.axes([0.50, 0.25, 0.25, 0.03], facecolor=axcolor)
ax_T = plt.axes([0.50, 0.20, 0.25, 0.03], facecolor=axcolor)
ax_rho = plt.axes([0.50, 0.15, 0.25, 0.03], facecolor=axcolor)
ax_amp = plt.axes([0.50, 0.10, 0.25, 0.03], facecolor=axcolor)

# 建立輸入框軸
ax_box_n = plt.axes([0.82, 0.25, 0.10, 0.03])
ax_box_T = plt.axes([0.82, 0.20, 0.10, 0.03])
ax_box_rho = plt.axes([0.82, 0.15, 0.10, 0.03])
ax_box_amp = plt.axes([0.82, 0.10, 0.10, 0.03])

# 建立拉桿 (包含單位與範圍說明)
s_n = Slider(ax_n, 'Harmonic n (1-10)', 1, 10, valinit=init_n, valstep=1)
s_T = Slider(ax_T, 'Tension T [N] (10-500)', 10.0, 500.0, valinit=init_T)
s_rho = Slider(ax_rho, 'Linear Density rho [kg/m] (0.1-5.0)', 0.1, 5.0, valinit=init_rho)
s_amp = Slider(ax_amp, 'Amplitude A [m] (0.1-2.0)', 0.1, 2.0, valinit=init_amp)

# 建立輸入框
text_box_n = TextBox(ax_box_n, '', initial=str(init_n))
text_box_T = TextBox(ax_box_T, '', initial=str(init_T))
text_box_rho = TextBox(ax_box_rho, '', initial=str(init_rho))
text_box_amp = TextBox(ax_box_amp, '', initial=str(init_amp))

# 輸入框回調函數
def submit_n(text):
    try:
        val = int(text)
        if 1 <= val <= 10:
            s_n.set_val(val)
        else:
            messagebox.showerror("Input Error", f"Harmonic n must be between 1 and 10.\nYou entered: {text}")
            text_box_n.set_val(str(int(s_n.val))) # Reset to current slider value
    except ValueError:
        messagebox.showerror("Input Error", f"Invalid input for Harmonic n.\nPlease enter an integer.\nYou entered: {text}")
        text_box_n.set_val(str(int(s_n.val)))

def submit_T(text):
    try:
        val = float(text)
        if 10.0 <= val <= 500.0:
            s_T.set_val(val)
        else:
            messagebox.showerror("Input Error", f"Tension T must be between 10.0 and 500.0.\nYou entered: {text}")
            text_box_T.set_val(str(s_T.val))
    except ValueError:
        messagebox.showerror("Input Error", f"Invalid input for Tension T.\nPlease enter a number.\nYou entered: {text}")
        text_box_T.set_val(str(s_T.val))

def submit_rho(text):
    try:
        val = float(text)
        if 0.1 <= val <= 5.0:
            s_rho.set_val(val)
        else:
            messagebox.showerror("Input Error", f"Linear Density rho must be between 0.1 and 5.0.\nYou entered: {text}")
            text_box_rho.set_val(str(s_rho.val))
    except ValueError:
        messagebox.showerror("Input Error", f"Invalid input for Linear Density rho.\nPlease enter a number.\nYou entered: {text}")
        text_box_rho.set_val(str(s_rho.val))

def submit_amp(text):
    try:
        val = float(text)
        if 0.1 <= val <= 2.0:
            s_amp.set_val(val)
        else:
            messagebox.showerror("Input Error", f"Amplitude A must be between 0.1 and 2.0.\nYou entered: {text}")
            text_box_amp.set_val(str(s_amp.val))
    except ValueError:
        messagebox.showerror("Input Error", f"Invalid input for Amplitude A.\nPlease enter a number.\nYou entered: {text}")
        text_box_amp.set_val(str(s_amp.val))

text_box_n.on_submit(submit_n)
text_box_T.on_submit(submit_T)
text_box_rho.on_submit(submit_rho)
text_box_amp.on_submit(submit_amp)

# 拉桿回調函數 (同步更新輸入框)
def update_text_n(val):
    text_box_n.set_val(str(int(val)))
def update_text_T(val):
    text_box_T.set_val(f"{val:.1f}")
def update_text_rho(val):
    text_box_rho.set_val(f"{val:.2f}")
def update_text_amp(val):
    text_box_amp.set_val(f"{val:.2f}")

s_n.on_changed(update_text_n)
s_T.on_changed(update_text_T)
s_rho.on_changed(update_text_rho)
s_amp.on_changed(update_text_amp)

# 建立資訊顯示軸 (位於拉桿左側)
ax_info = plt.axes([0.02, 0.05, 0.35, 0.3]) # 加大顯示區域
ax_info.axis('off')

# 顯示參數值的文字
text_params = ax_info.text(0.0, 1.0, '', transform=ax_info.transAxes, verticalalignment='top', fontsize=12) # 加大字體並移除框線

def update(frame):
    # 讀取拉桿數值
    n = int(s_n.val)
    T = s_T.val
    rho = s_rho.val
    A = s_amp.val
    
    # 物理計算
    # v = sqrt(T / rho)  (Eq 2)
    v = np.sqrt(T / rho)
    
    # lambda = 2L / n    (Eq 5)
    wavelength = 2 * L / n
    
    # f = v / lambda     (Eq 1)
    f = v / wavelength
    
    # k = 2pi / lambda
    k = 2 * np.pi / wavelength
    
    # omega = 2pi * f
    omega = 2 * np.pi * f
    
    # 時間 t (模擬時間流逝)
    t = frame / 100.0 
    
    # 兩個行進波
    y1 = A * np.sin(k * x - omega * t) # 向右
    y2 = A * np.sin(k * x + omega * t) # 向左
    
    # 駐波 (疊加)
    y = y1 + y2
    
    line.set_ydata(y)
    line1.set_ydata(y1)
    line2.set_ydata(y2)
    
    # 更新文字顯示
    info_text = (
        fr'Time: $t = {t:.2f}$ s' + '\n' +
        fr'Harmonic Mode: $n = {n}$' + '\n' +
        fr'String Length: $L = {L:.1f}$ m' + '\n' +
        fr'Tension: $T = {T:.1f}$ N' + '\n' +
        fr'Density: $\rho = {rho:.2f}$ kg/m' + '\n' +
        '----------------' + '\n' +
        fr'Wave Speed: $v = \sqrt{{T/\rho}} = {v:.2f}$ m/s' + '\n' +
        fr'Wavelength: $\lambda = 2L/n = {wavelength:.2f}$ m' + '\n' +
        fr'Frequency: $f = v/\lambda = {f:.2f}$ Hz'
    )
    text_params.set_text(info_text)
    
    # Update curve (because n or rho might have changed)
    new_f_vals = get_f_curve(n, rho, T_vals)
    line_curve.set_ydata(new_f_vals)
    
    # Update point
    point_curr.set_data([T], [f])
    
    # Rescale y axis for ax_rel
    ax_rel.set_ylim(0, max(new_f_vals) * 1.1)
    
    return line, line1, line2, text_params, line_curve, point_curr

# 建立動畫
# interval=10ms 對應 100fps
ani = animation.FuncAnimation(fig, update, interval=10, blit=True, cache_frame_data=False)

# 建立按鈕軸
ax_import = plt.axes([0.54, 0.025, 0.1, 0.04])
ax_start = plt.axes([0.65, 0.025, 0.1, 0.04])
ax_stop = plt.axes([0.76, 0.025, 0.1, 0.04])
ax_quit = plt.axes([0.87, 0.025, 0.1, 0.04])

# 建立按鈕
btn_import = Button(ax_import, 'Import', color='lightblue', hovercolor='0.975')
btn_start = Button(ax_start, 'Start', color='lightgreen', hovercolor='0.975')
btn_stop = Button(ax_stop, 'Stop', color='salmon', hovercolor='0.975')
btn_quit = Button(ax_quit, 'Quit', color='lightgray', hovercolor='0.975')

def start_anim(event):
    line.set_animated(True)
    line1.set_animated(True)
    line2.set_animated(True)
    text_params.set_animated(True)
    line_curve.set_animated(True)
    point_curr.set_animated(True)
    ani.event_source.start()

def stop_anim(event):
    ani.event_source.stop()
    line.set_animated(False)
    line1.set_animated(False)
    line2.set_animated(False)
    text_params.set_animated(False)
    line_curve.set_animated(False)
    point_curr.set_animated(False)
    fig.canvas.draw()

def import_params(event):
    filename = 'standing_wave_params.json'
    if not os.path.exists(filename):
        messagebox.showwarning("Import Failed", f"File '{filename}' not found.\nPlease save parameters first by quitting the simulation.")
        return
    
    try:
        with open(filename, 'r') as f:
            params = json.load(f)
            
        # Update sliders (this will trigger on_changed and update the plot)
        if 'n' in params: s_n.set_val(params['n'])
        if 'T' in params: s_T.set_val(params['T'])
        if 'rho' in params: s_rho.set_val(params['rho'])
        if 'A' in params: s_amp.set_val(params['A'])
        
        messagebox.showinfo("Import Success", f"Parameters loaded from '{filename}'")
        
    except Exception as e:
        messagebox.showerror("Import Error", f"Failed to load parameters:\n{str(e)}")

def quit_sim(event):
    # Get current values
    n = int(s_n.val)
    T = s_T.val
    rho = s_rho.val
    A = s_amp.val
    
    # Calculate derived values
    v = np.sqrt(T / rho)
    wavelength = 2 * L / n
    f = v / wavelength
    
    # Save to JSON
    params = {
        'n': n,
        'T': T,
        'rho': rho,
        'A': A
    }
    filename = 'standing_wave_params.json'
    try:
        with open(filename, 'w') as json_file:
            json.dump(params, json_file, indent=4)
        print(f"Parameters saved to {filename}")
    except Exception as e:
        print(f"Failed to save parameters: {e}")
    
    print("\n" + "="*30)
    print("Simulation Stopped by User")
    print("="*30)
    print(f"Recorded Parameters:")
    print(f"  Harmonic Mode (n)    : {n}")
    print(f"  Tension (T)          : {T:.2f} N")
    print(f"  Linear Density (rho) : {rho:.2f} kg/m")
    print(f"  Amplitude (A)        : {A:.2f} m")
    print("-" * 30)
    print(f"  Wave Speed (v)       : {v:.2f} m/s")
    print(f"  Wavelength (lambda)  : {wavelength:.2f} m")
    print(f"  Frequency (f)        : {f:.2f} Hz")
    print("="*30 + "\n")
    
    plt.close(fig)

btn_import.on_clicked(import_params)
btn_start.on_clicked(start_anim)
btn_stop.on_clicked(stop_anim)
btn_quit.on_clicked(quit_sim)

plt.show()
