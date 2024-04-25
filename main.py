import cv2
import numpy as np

# Загрузка изображения
img = cv2.imread('color_text.jpg')

# Преобразование изображения в оттенки серого
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Применение оператора Canny для обнаружения контуров
edges = cv2.Canny(gray, 25, 53)

# Нахождение контуров текста
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Сортировка контуров по вертикальной координате (y)
contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[1])

# Создание черного фона
black_bg = np.zeros_like(img)

cv2.drawContours(black_bg, contours[:241], -1, color=((0,255,0)), thickness=1)
cv2.drawContours(black_bg, contours[242:572], -1, color=((0,0,255)), thickness=1)
cv2.drawContours(black_bg, contours[572:750], -1, color=((255,0,0)), thickness=1)


# Показываем и сохраняем результат
cv2.imshow('Contours', black_bg)
cv2.imwrite('contours_output.jpg', black_bg)
cv2.waitKey(0)
cv2.destroyAllWindows()
