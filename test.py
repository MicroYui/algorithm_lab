import tkinter as tk
import threading
import time
import random
import sys
import os

# 尝试导入 winsound (仅 Windows) 以获得更好的提示音
try:
    import winsound
    SOUND_AVAILABLE = True
except ImportError:
    SOUND_AVAILABLE = False
    print("提示：'winsound' 模块未找到。将使用简单的打印提示音（可能无声）。")
    print("      如果在 Windows 上运行，可以尝试安装 winsound。")

# --- 全局变量 ---
timer_thread = None      # 用于存储计时器线程
stop_event = threading.Event() # 用于通知线程停止的事件

# --- 功能函数 ---

def play_alert_sound():
    """根据操作系统播放提示音"""
    print("响铃！") # 无论如何都打印提示
    if SOUND_AVAILABLE and sys.platform == 'win32':
        # 使用 winsound 在 Windows 上发出哔声 (频率 1000Hz, 持续 500ms)
        winsound.Beep(1000, 500)
    else:
        # 尝试打印 BEL 字符 (ASCII 7)，在某些终端会发出哔声
        # 在许多现代系统或 IDE 中可能无效
        print('\a', flush=True)
        # 可以考虑添加其他平台的播放命令，例如 macOS 的 'afplay' 或 Linux 的 'aplay'/'paplay'
        # if sys.platform == 'darwin': # macOS
        #     os.system('afplay /System/Library/Sounds/Ping.aiff') # 需要一个声音文件
        # elif 'linux' in sys.platform: # Linux
        #     # 尝试 aplay (需要 ALSA utils 和声音文件)
        #     # os.system('aplay /path/to/sound.wav')
        #     # 或者尝试 paplay (需要 PulseAudio utils 和声音文件)
        #     # os.system('paplay /path/to/sound.wav')
        #     pass # 如果没有简单方法，则仅打印


def timer_loop():
    """在后台运行的计时器循环"""
    global stop_event
    print("计时器线程已启动。")

    while not stop_event.is_set():
        try:
            # 1. 等待随机时间 (3-5 分钟)
            wait_seconds = random.uniform(3 * 60, 5 * 60)
            # wait_seconds = random.uniform(5, 10) # 测试用：改为5-10秒
            print(f"计时器：等待 {wait_seconds:.2f} 秒...")

            # 使用 event.wait() 代替 time.sleep()，这样可以更快地响应停止信号
            # wait() 返回 True 如果事件被设置 (即请求停止), 返回 False 如果超时
            if stop_event.wait(timeout=wait_seconds):
                print("计时器：在等待期间收到停止信号，退出。")
                break # 如果在等待期间被要求停止，则退出循环

            if stop_event.is_set(): # 再次检查以防万一
                 print("计时器：等待结束后检测到停止信号，退出。")
                 break

            # 2. 第一次响铃
            print("计时器：第一次响铃时间到！")
            play_alert_sound()

            # 3. 等待 10 秒
            print("计时器：等待 10 秒...")
            if stop_event.wait(timeout=10):
                print("计时器：在10秒等待期间收到停止信号，退出。")
                break

            if stop_event.is_set(): # 再次检查
                 print("计时器：10秒等待结束后检测到停止信号，退出。")
                 break

            # 4. 第二次响铃
            print("计时器：第二次响铃时间到！")
            play_alert_sound()

            print("-" * 20) # 分隔符，表示一次循环结束

        except Exception as e:
            print(f"计时器线程出错: {e}")
            break # 如果发生错误，也停止循环

    print("计时器线程已停止。")

def start_timer():
    """启动计时器线程"""
    global timer_thread, stop_event

    # 如果计时器已在运行，则不执行任何操作 (或者先停止旧的？这里选择不重复启动)
    if timer_thread is not None and timer_thread.is_alive():
        print("计时器已经在运行中。")
        return

    print("点击了开始按钮。")
    stop_event.clear() # 重置停止事件标志
    status_label.config(text="状态：运行中...")
    start_button.config(state=tk.DISABLED) # 禁用开始按钮，防止重复点击
    stop_button.config(state=tk.NORMAL)   # 启用停止按钮

    # 创建并启动新的计时器线程
    # 设置为守护线程(daemon=True)，这样主程序退出时该线程也会强制退出
    timer_thread = threading.Thread(target=timer_loop, daemon=True)
    timer_thread.start()

def stop_program():
    """设置停止事件并关闭程序"""
    global stop_event, root
    print("点击了停止按钮。")
    stop_event.set() # 发送停止信号给计时器线程
    status_label.config(text="状态：正在停止...")
    stop_button.config(state=tk.DISABLED) # 禁用停止按钮

    # 可以选择给线程一点时间来响应停止信号，但因为它是守护线程，
    # destroy() 会最终结束它。直接关闭窗口通常也可以。
    # time.sleep(0.1) # 短暂等待（可选）

    print("关闭应用程序窗口。")
    root.destroy() # 关闭主窗口，这将结束 tk.mainloop()

# --- GUI 设置 ---
root = tk.Tk()
root.title("随机计时器")
root.geometry("300x150") # 设置窗口大小

# 状态标签
status_label = tk.Label(root, text="状态：未开始", font=("Arial", 12))
status_label.pack(pady=10)

# 按钮框架
button_frame = tk.Frame(root)
button_frame.pack(pady=10)

# 开始按钮
start_button = tk.Button(button_frame, text="开始", command=start_timer, width=10, height=2)
start_button.pack(side=tk.LEFT, padx=10)

# 停止按钮
stop_button = tk.Button(button_frame, text="停止", command=stop_program, width=10, height=2)
stop_button.pack(side=tk.LEFT, padx=10)
stop_button.config(state=tk.DISABLED) # 初始时停止按钮不可用

# --- 运行 GUI ---
root.mainloop()

# mainloop 结束后，程序会在这里退出
print("应用程序已关闭。")