import laspy
import numpy as np


las = laspy.read("Univer2019.las")
# Определяем point format из оригинала
original_fmt = las.header.point_format.id

# LAS 1.2 поддерживает форматы 0–5
header = laspy.LasHeader(point_format=original_fmt, version="1.2")

unlabeled_las = laspy.LasData(header)

# Копируем только координаты
unlabeled_las.x = las.x
unlabeled_las.y = las.y
unlabeled_las.z = las.z

# НЕ копируем classification и другие атрибуты

# Сохраняем
unlabeled_las.write("unlabeled.las")
print("✅ Создан файл unlabeled.las (версия 1.2, без меток)")