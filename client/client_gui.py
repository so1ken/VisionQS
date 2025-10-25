import cv2 # либра OpenCV для камеры
import tkinter as tk # GUI фреймворк Python
import os;
from PIL import Image, ImageTk # Конвертация OpenCV матрицу в объект, который Tkinter сможет вывести 
import requests
import threading

# Открываем камеру 0 = встроенная
cap = cv2.VideoCapture(0)

# установим разрешение камеры (моя камера 640x480)
CAM_W, CAM_H = 640, 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

# Сосздаем окно приложения, root - главное окно / l = lable
root = tk.Tk()
root.title("VisionQC")

l = tk.Label(root)
l.pack()

status_label = tk.Label(root, text="Ready")
status_label.pack()

# Создаем папку для фотографий захвата
save_folder = 'captures'
os.makedirs(save_folder, exist_ok=True)

counter = len([name for name in os.listdir(save_folder) if os.path.isfile(os.path.join(save_folder, name))]) + 1 # Счетчик в имени фотографии продукта

SERVER_URL = 'http://localhost:8000/analyze'  # адрес сервера (меняй при необходимости)
REQUEST_TIMEOUT = 6  # секунда таймаута для запросов

def update_status(text):
    root.after(0, lambda: status_label.config(text=text))

# Фоновая отправка файла на сервер (чтобы GUI не вис)
def send_to_server_async(filepath, filename):
    def job():
        update_status("Sending...")
        try:
            with open(filepath, 'rb') as f:
                files = {'image': (filename, f, 'image/jpeg')}
                resp = requests.post(SERVER_URL, files=files, timeout=REQUEST_TIMEOUT)
            # обработка ответа
            try:
                data = resp.json()
            except Exception:
                update_status(f"Server returned {resp.status_code}")
                return
            if resp.status_code == 200:
                res = data.get('result', 'unknown')
                score = data.get('score', None)
                if score is not None:
                    update_status(f"Received: {res} (score={score:.6f})")
                else:
                    update_status(f"Received: {res}")
            else:
                update_status(f"Server error: {data}")
        except requests.exceptions.RequestException as e:
            update_status(f"Network error: {e}")
        except Exception as e:
            update_status(f"Error: {e}")

    threading.Thread(target=job, daemon=True).start()

# Функция обновления кадра
def update_frame():
    ret, frame = cap.read() #ret, frame = cap.read() — считываем один кадр; ret = True, если успешно
    if ret:
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # OpenCV хранит цвет в порядке BGR — нужно конвертировать в RGB перед показом в Tkinter
        im = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=im)

        l.imgtk = imgtk
        l.configure(image=imgtk) # Отображаем 

    l.after(10, update_frame) # планируем следующий вызов через 10 мс

# Функция захват, вызываем когда нажимаем на кнопку
def capture():
    global counter
    ret, frame = cap.read() #ret, frame = cap.read() — считываем один кадр; ret = True, если успешно
    if ret:
        filename = f"Analiz_{counter}.jpg"
        filepath = os.path.join(save_folder, filename)
        cv2.imwrite(filepath, frame)
        counter += 1

        send_to_server_async(filepath, filename)
    else:
        update_status("Camera read failed")

# Создаем кнопку (В нашем окне root, добавляем на нее текст, и задаем ей команду - нашу функцию)
btn = tk.Button(root, text="Photo", command=capture) 
btn.pack()

def on_close():
    try:
        cap.release()
    except:
        pass
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_close)


update_frame()
root.mainloop()

cap.release()
cv2.destroyAllWindows()