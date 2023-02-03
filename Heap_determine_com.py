import cv2
import numpy as np
import serial
import time
import keyboard as kb

# сжимаем картинку на scale_percent %
def img_zip(img, scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dsize = (width, height)
    output = cv2.resize(img, dsize)
    return output

def nothing(*arg):
    pass

# восстанавливаем балланс белого на кадре
def white_balance(img):
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result

cv2.namedWindow("result") # создаем главное окно
cv2.namedWindow("settings") # создаем окно подстройки

# Устанавливаем величину порога контуров
cv2.createTrackbar('threshold', 'settings', 0, 255, nothing)
cv2.setTrackbarPos('threshold', 'settings', 180)    

# координаты образки экрана
x_min = 550
x_max = 950
y_min = 245
y_max = 485
# Палитра HSV для зеленого цвета - обведенного изображения
green_hsv_min = np.array((20, 175, 30), np.uint8)
green_hsv_max = np.array((175, 255, 255), np.uint8)
# Линия уровня
line_thickness = 2
# Открываем COM-порт для вывода сигнала срабатывания
ser1 = serial.Serial('COM3', timeout=1)

# Работа с видео
cap = cv2.VideoCapture("SCHOM9.mp4")
while cap.isOpened():
    try:
        ret, image = cap.read()
        image_crop = image[y_min:y_max, x_min:x_max]         # Обрезаем нужную часть изображения, сперва высота, потом ширина
    except:
        print('Video is finished')
        break
    
    image_crop = white_balance(image_crop)                          # восстанавливаем балланс белого на кадре
    monochrome_image = cv2.cvtColor(image_crop,cv2.COLOR_BGR2GRAY)  # делаем кадр черно-белым
    thresh = cv2.getTrackbarPos('threshold', 'settings')            # смотрим текщие настройки порога контуров
   
    ret, thresh_image = cv2.threshold(monochrome_image, thresh, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(image=thresh_image.copy(), mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE) #ищем контуры на кадре в выделенной области
    image_copy=image_crop.copy()
    cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0,255,0), thickness=2, lineType=cv2.LINE_AA)    # рисуем контуры зеленым цветом

# Теперь накладываем фильтр для зеленоого цвета:
    green_hsv = cv2.cvtColor(image_copy, cv2.COLOR_BGR2HSV )
    my_hsv = cv2.inRange(green_hsv, green_hsv_min, green_hsv_max)
# вычисляем моменты изображения
    moments = cv2.moments(my_hsv, 1)
    dM01 = moments['m01']
    dM10 = moments['m10']
    dArea = moments['m00']
# будем реагировать только на те моменты, которые содержать больше 100 пикселей
    if dArea > 100:
        x = int(dM10 / dArea)
        y = int(dM01 / dArea)
    cv2.line(image_copy, (x-150, y-90), (x+150, y-90), (0, 0, 255), thickness=line_thickness)   # Линия уровня. Немного смещена по y

# Вставляем анализируемую область в кадр
    image[y_min:y_max, x_min:x_max] = image_copy
    # Рисуем слева шкалу:
    cv2.line(image, (150, 0), (150, 120), (0, 0, 255), thickness=line_thickness)
    cv2.line(image, (150, 121), (150, 240), (0, 255, 255), thickness=line_thickness)
    cv2.line(image, (150, 241), (150, 350), (0, 255, 0), thickness=line_thickness)
    # Выводим уровень шкалы числом
    cv2.putText(image, str(y), (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    # Если порог высоты превышен, то выдаем сигнал на COM-порт
    if y<130:
        ser1.rts =  1
    else:
        ser1.rts = 0

# Выводим итог:
    cv2.imshow("Only_threshold", image)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
ser1.close()
cv2.destroyAllWindows()