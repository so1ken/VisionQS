# client_gui.py
"""
VisionQC client — modern GUI with statistics + server controls
- statistics: total, good, bad, defect rate
- recent results list (last 10)
- start / stop / restart server buttons
- retrain button still present
- logs saved to logs/analysis_log.csv

Изменения:
- увеличено окно (1360x800)
- добавлена кнопка включения/выключения камеры сверху слева над превью
"""
import os
import re
import threading
import time
import subprocess
import sys
import csv
from collections import deque

import cv2
import customtkinter as ctk
from PIL import Image, ImageTk
import requests

# ---------------- CONFIG ----------------
SAVE_FOLDER = "captures"
FNAME_TEMPLATE = "Analiz_{}.jpg"
SERVER_URL = "http://localhost:8000/analyze"
CAM_WIDTH = 640
CAM_HEIGHT = 480
REQUEST_TIMEOUT = 8
CAM_POLL_DELAY = 0.01
LOGS_DIR = "logs"
LOG_CSV = os.path.join(LOGS_DIR, "analysis_log.csv")

# clear proxy env vars to avoid VPN/proxy interfering with localhost requests
for k in ("HTTP_PROXY","HTTPS_PROXY","http_proxy","https_proxy"):
    os.environ.pop(k, None)

os.makedirs(SAVE_FOLDER, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# ensure CSV header exists
if not os.path.exists(LOG_CSV):
    with open(LOG_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp","filename","result","score"])

# compute starting counter based on existing files Analiz_#.jpg
def _init_counter():
    existing = os.listdir(SAVE_FOLDER)
    nums = []
    pattern = re.compile(r"Analiz_(\d+)\.jpg$")
    for name in existing:
        m = pattern.match(name)
        if m:
            try:
                nums.append(int(m.group(1)))
            except:
                pass
    return max(nums) + 1 if nums else 1

counter = _init_counter()

# ---------------- Camera (background reader) ----------------
try:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
except Exception:
    cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

frame_lock = threading.Lock()
latest_frame = None
camera_running = True  # thread life control
camera_active = True   # camera on/off toggle (True = capturing)

def camera_reader():
    global latest_frame, camera_running, camera_active
    while camera_running:
        if camera_active:
            ret, frame = cap.read()
            if ret:
                with frame_lock:
                    latest_frame = frame.copy()
            else:
                # couldn't read frame; small sleep to avoid busy loop
                time.sleep(0.05)
        else:
            # camera is disabled — clear preview frame (so UI shows blank)
            with frame_lock:
                latest_frame = None
            time.sleep(0.1)
        time.sleep(CAM_POLL_DELAY)

cam_thread = threading.Thread(target=camera_reader, daemon=True)
cam_thread.start()

# ---------------- Statistics ----------------
total_analyzed = 0
good_count = 0
bad_count = 0
recent_results = deque(maxlen=10)
stats_lock = threading.Lock()

# ---------------- UI init ----------------
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.title("VisionQC - Client")
# увеличил размер окна, чтобы 640x480 превью влезало с запасом
app.geometry("1360x800")

app.grid_columnconfigure(0, weight=3)
app.grid_columnconfigure(1, weight=1)
app.grid_rowconfigure(0, weight=1)

# left: preview (with top-left camera toggle)
preview_frame = ctk.CTkFrame(app)
preview_frame.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)
preview_frame.grid_rowconfigure(0, weight=0)  # controls row
preview_frame.grid_rowconfigure(1, weight=1)  # preview row
preview_frame.grid_rowconfigure(2, weight=0)  # status row
preview_frame.grid_columnconfigure(0, weight=1)

# top controls over preview (left-top)
preview_controls = ctk.CTkFrame(preview_frame)
preview_controls.grid(row=0, column=0, sticky="nw", padx=6, pady=(6,0))

# camera toggle button
def toggle_camera():
    global camera_active
    camera_active = not camera_active
    if camera_active:
        btn_cam_toggle.configure(text="Camera: ON")
        update_status("Camera enabled")
    else:
        btn_cam_toggle.configure(text="Camera: OFF")
        update_status("Camera disabled")
    # latest_frame cleared by camera_reader when camera_active == False

btn_cam_toggle = ctk.CTkButton(preview_controls, text="Camera: ON", width=120, command=toggle_camera)
btn_cam_toggle.grid(row=0, column=0, padx=(0,6), pady=4)

# preview label (the video)
preview_label = ctk.CTkLabel(preview_frame, text="")
preview_label.grid(row=1, column=0, sticky="nsew", padx=8, pady=8)

# status label under preview
status_label = ctk.CTkLabel(preview_frame, text="Ready")
status_label.grid(row=2, column=0, sticky="ew", padx=8, pady=(0,8))

# right: controls / stats / console
right_frame = ctk.CTkFrame(app)
right_frame.grid(row=0, column=1, sticky="nsew", padx=12, pady=12)
right_frame.grid_rowconfigure(0, weight=0)
right_frame.grid_rowconfigure(1, weight=0)
right_frame.grid_rowconfigure(2, weight=1)
right_frame.grid_columnconfigure(0, weight=1)

# controls
controls_frame = ctk.CTkFrame(right_frame)
controls_frame.grid(row=0, column=0, sticky="ew", padx=8, pady=8)
controls_frame.grid_columnconfigure((0,1), weight=1)

# stats
stats_frame = ctk.CTkFrame(right_frame)
stats_frame.grid(row=1, column=0, sticky="ew", padx=8, pady=8)
stats_frame.grid_columnconfigure(0, weight=1)

# console
console_frame = ctk.CTkFrame(right_frame)
console_frame.grid(row=2, column=0, sticky="nsew", padx=8, pady=(6,8))
console_frame.grid_rowconfigure(0, weight=1)
console_frame.grid_columnconfigure(0, weight=1)

# console widget
try:
    console = ctk.CTkTextbox(console_frame, wrap="word", state="normal")
    console.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
    console.insert("0.0", "Console initialized...\n")
    console.configure(state="disabled")
    def console_write(text):
        console.configure(state="normal")
        console.insert("end", text + "\n")
        console.see("end")
        console.configure(state="disabled")
except Exception:
    import tkinter.scrolledtext as st
    console = st.ScrolledText(console_frame, wrap="word", state="disabled")
    console.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
    def console_write(text):
        console.configure(state="normal")
        console.insert("end", text + "\n")
        console.see("end")
        console.configure(state="disabled")

# ---------------- Controls logic ----------------
last_saved_path = None
last_saved_name = None

server_proc = None
train_proc = None
process_lock = threading.Lock()

def update_status(text):
    status_label.configure(text=text)
    console_write(f"[{time.strftime('%H:%M:%S')}] {text}")

def save_log_row(timestamp, filename, result, score):
    try:
        with open(LOG_CSV, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, filename, result, score])
    except Exception as e:
        console_write(f"Failed to write log: {e}")

def send_file_background(filepath, filename):
    def job():
        global total_analyzed, good_count, bad_count
        update_status("Sending...")
        try:
            with open(filepath, "rb") as f:
                files = {"image": (filename, f, "image/jpeg")}
                resp = requests.post(SERVER_URL, files=files, timeout=REQUEST_TIMEOUT)
            console_write(f"HTTP {resp.status_code} from server")
            try:
                data = resp.json()
                console_write(f"Server JSON: {data}")
                res = data.get("result") or data.get("status") or data.get("note") or "unknown"
                score = data.get("score") if isinstance(data.get("score"), (int, float)) else None
                with stats_lock:
                    total_analyzed += 1
                    if str(res).lower() in ('good','ok','received'):
                        good_count += 1
                    else:
                        bad_count += 1
                    recent_results.appendleft((time.strftime('%Y-%m-%d %H:%M:%S'), filename, res, score))
                update_stats_widgets()
                save_log_row(time.strftime('%Y-%m-%d %H:%M:%S'), filename, res, score)
                update_status(f"Received: {res}")
            except Exception as e:
                console_write(f"Failed to parse JSON: {e}")
                console_write(f"Text: {resp.text}")
                update_status(f"Server returned status {resp.status_code}")
        except requests.exceptions.RequestException as e:
            console_write(f"Network error: {e}")
            update_status("Network error")
        except Exception as e:
            console_write(f"Error sending file: {e}")
            update_status("Send error")
    threading.Thread(target=job, daemon=True).start()

def do_capture():
    global counter, last_saved_path, last_saved_name
    with frame_lock:
        frame = latest_frame.copy() if latest_frame is not None else None
    if frame is None:
        update_status("No frame available")
        return
    filename = FNAME_TEMPLATE.format(counter)
    filepath = os.path.join(SAVE_FOLDER, filename)
    ok = cv2.imwrite(filepath, frame)
    if not ok:
        update_status("Failed to save image")
        return
    counter += 1
    last_saved_path = filepath
    last_saved_name = filename
    update_status(f"Saved: {filepath}")
    send_file_background(filepath, filename)

# Theme switch callback (switch widget)
def theme_switch_cb():
    mode = "Dark" if theme_switch.get() else "Light"
    ctk.set_appearance_mode(mode)
    update_status(f"Theme: {mode}")

# Process streaming helper
def stream_process_output(proc, label_prefix="proc"):
    try:
        if proc.stdout is None:
            return
        for raw in proc.stdout:
            if not raw:
                break
            try:
                s = raw.decode(errors="ignore").rstrip()
            except AttributeError:
                s = str(raw).rstrip()
            console_write(f"[{label_prefix}] {s}")
    except Exception as e:
        console_write(f"[{label_prefix}] stream error: {e}")

# server control: start/stop/restart
def start_server():
    global server_proc
    with process_lock:
        if server_proc is not None and server_proc.poll() is None:
            update_status("Server already running")
            return
        update_status("Starting server...")
        cmd = [sys.executable, os.path.join("server", "server_app.py")]
        try:
            server_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            threading.Thread(target=stream_process_output, args=(server_proc, "server"), daemon=True).start()
            update_status(f"Server started (pid={server_proc.pid})")
        except Exception as e:
            update_status(f"Failed to start server: {e}")
            console_write(f"Exception: {e}")

def stop_server():
    global server_proc
    with process_lock:
        if server_proc is None or server_proc.poll() is not None:
            update_status("Server is not running")
            return
        update_status("Stopping server...")
        try:
            server_proc.terminate()
            try:
                server_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_proc.kill()
            update_status("Server stopped")
            server_proc = None
        except Exception as e:
            update_status(f"Failed to stop server: {e}")
            console_write(f"Exception: {e}")

def restart_server():
    update_status("Restarting server...")
    stop_server()
    time.sleep(0.5)
    start_server()

# retrain trigger
def retrain_model():
    global train_proc
    with process_lock:
        if train_proc is not None and train_proc.poll() is None:
            update_status("Training already running")
            return
        update_status("Starting training...")
        cmd = [sys.executable, os.path.join("train", "train.py"),
               "--data-dir", "data",
               "--out-model-dir", "server/model",
               "--epochs", "5"]
        try:
            train_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            threading.Thread(target=stream_process_output, args=(train_proc, "train"), daemon=True).start()
            update_status(f"Training started (pid={train_proc.pid})")
        except Exception as e:
            update_status(f"Failed to start training: {e}")
            console_write(f"Exception: {e}")

# ---------------- Stats widgets ----------------
lbl_total = ctk.CTkLabel(stats_frame, text="Total: 0")
lbl_total.grid(row=0, column=0, sticky="w", padx=6, pady=2)

lbl_good = ctk.CTkLabel(stats_frame, text="Good: 0")
lbl_good.grid(row=0, column=1, sticky="w", padx=6, pady=2)

lbl_bad = ctk.CTkLabel(stats_frame, text="Bad: 0")
lbl_bad.grid(row=0, column=2, sticky="w", padx=6, pady=2)

lbl_rate = ctk.CTkLabel(stats_frame, text="Defect rate: 0%")
lbl_rate.grid(row=1, column=0, columnspan=3, sticky="w", padx=6, pady=2)

recent_frame = ctk.CTkFrame(stats_frame)
recent_frame.grid(row=2, column=0, columnspan=3, sticky="ew", padx=6, pady=6)
recent_frame.grid_columnconfigure(0, weight=1)

try:
    recent_box = ctk.CTkTextbox(recent_frame, height=8, state="normal")
    recent_box.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
    recent_box.insert("0.0", "Recent results...\n")
    recent_box.configure(state="disabled")
    def update_recent_box():
        recent_box.configure(state="normal")
        recent_box.delete("0.0", "end")
        for entry in recent_results:
            ts, fn, res, sc = entry
            recent_box.insert("end", f"{ts} | {fn} | {res} | {sc}\n")
        recent_box.configure(state="disabled")
except Exception:
    import tkinter as tk
    recent_box = tk.Text(recent_frame, height=8, state="disabled")
    recent_box.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
    def update_recent_box():
        recent_box.configure(state="normal")
        recent_box.delete("1.0", "end")
        for entry in recent_results:
            ts, fn, res, sc = entry
            recent_box.insert("end", f"{ts} | {fn} | {res} | {sc}\n")
        recent_box.configure(state="disabled")

def update_stats_widgets():
    with stats_lock:
        lbl_total.configure(text=f"Total: {total_analyzed}")
        lbl_good.configure(text=f"Good: {good_count}")
        lbl_bad.configure(text=f"Bad: {bad_count}")
        rate = (bad_count / total_analyzed * 100) if total_analyzed > 0 else 0
        lbl_rate.configure(text=f"Defect rate: {rate:.1f}%")
    update_recent_box()

# ---------------- Controls layout ----------------
btn_capture = ctk.CTkButton(controls_frame, text="Photo", command=do_capture)
btn_capture.grid(row=0, column=0, sticky="ew", padx=6, pady=6)

theme_switch = ctk.CTkSwitch(controls_frame, text="Dark theme", command=theme_switch_cb)
theme_switch.grid(row=0, column=1, sticky="ew", padx=6, pady=6)

server_controls = ctk.CTkFrame(controls_frame)
server_controls.grid(row=1, column=0, columnspan=2, sticky="ew", padx=6, pady=6)
server_controls.grid_columnconfigure((0,1,2), weight=1)

btn_start_server = ctk.CTkButton(server_controls, text="Start", command=start_server)
btn_start_server.grid(row=0, column=0, sticky="ew", padx=4)

btn_stop_server = ctk.CTkButton(server_controls, text="Stop", fg_color="#ff5c5c", hover_color="#ff7b7b", command=stop_server)
btn_stop_server.grid(row=0, column=1, sticky="ew", padx=4)

btn_restart_server = ctk.CTkButton(server_controls, text="Restart", command=restart_server)
btn_restart_server.grid(row=0, column=2, sticky="ew", padx=4)

btn_retrain = ctk.CTkButton(controls_frame, text="Retrain Model", command=retrain_model)
btn_retrain.grid(row=2, column=0, columnspan=2, sticky="ew", padx=6, pady=6)

# ---------------- Preview loop ----------------
preview_img_ref = None

def update_preview():
    global preview_img_ref
    with frame_lock:
        frame = latest_frame.copy() if latest_frame is not None else None
    if frame is not None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(pil)
        preview_label.configure(image=imgtk)
        preview_img_ref = imgtk
    else:
        # clear preview image when no frame
        preview_label.configure(image="")
        preview_img_ref = None
    app.after(30, update_preview)

# ---------------- Shutdown ----------------
def on_closing():
    global camera_running, server_proc, train_proc
    update_status("Shutting down...")
    camera_running = False
    time.sleep(0.05)
    try:
        cap.release()
    except:
        pass
    with process_lock:
        if train_proc is not None and train_proc.poll() is None:
            try:
                train_proc.terminate()
                console_write("Sent terminate to training process")
            except:
                pass
        if server_proc is not None and server_proc.poll() is None:
            try:
                server_proc.terminate()
                console_write("Sent terminate to server process")
            except:
                pass
    app.destroy()

app.protocol("WM_DELETE_WINDOW", on_closing)

# init
console_write("VisionQC Client started")
console_write(f"Save folder: {os.path.abspath(SAVE_FOLDER)}")
console_write(f"Server URL: {SERVER_URL}")
update_stats_widgets()
update_preview()
app.mainloop()
