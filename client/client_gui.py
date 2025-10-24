import cv2 # либра OpenCV для камеры
import tkinter as tk # GUI фреймворк Python
import os;
from PIL import Image, ImageTk # Конвертация OpenCV матрицу в объект, который Tkinter сможет вывести 

# Открываем камеру 0 = встроенная
cap = cv2.VideoCapture(0)

# Сосздаем окно приложения, root - главное окно / l = lable
root = tk.Tk()
root.title("VisionQC")

l = tk.Label(root)
l.pack()

# Создаем папку для фотографий захвата
save_folder = 'captures'
os.makedirs(save_folder, exist_ok=True)

counter = len([name for name in os.listdir(save_folder) if os.path.isfile(os.path.join(save_folder, name))]) + 1 # Счетчик в имени фотографии продукта

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

# Создаем кнопку (В нашем окне root, добавляем на нее текст, и задаем ей команду - нашу функцию)
btn = tk.Button(root, text="Photo", command=capture) 
btn.pack()

update_frame()
root.mainloop()

cap.release()
cv2.destroyAllWindows()