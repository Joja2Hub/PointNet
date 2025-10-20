import laspy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_las(las_file, max_points=50000, title="Point Cloud"):
    """Визуализация LAS файла"""
    print(f"📂 Загрузка {las_file}...")
    las = laspy.read(las_file)
    
    # Загрузка координат и классов
    points = np.vstack([las.x, las.y, las.z]).T
    
    try:
        classes = las.classification
        has_classes = True
    except:
        classes = np.zeros(len(points))
        has_classes = False
    
    # Сэмплирование для визуализации
    if len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        points = points[indices]
        classes = classes[indices]
    
    # Цветовая карта для классов
    colors_map = {
        0: [0.5, 0.5, 0.5],  # Не классифицировано - серый
        1: [0.6, 0.4, 0.2],  # Класс 1 - коричневый
        2: [0.3, 0.2, 0.1],  # Класс 2 - темно-коричневый  
        3: [0.1, 0.5, 0.1],  # Класс 3 - зеленый
        4: [0.0, 0.3, 0.0],  # Класс 4 - темно-зеленый
        5: [0.8, 0.8, 0.2],  # Класс 5 - желтый
        6: [0.9, 0.1, 0.1],  # Класс 6 - красный
        7: [0.1, 0.1, 0.9],  # Класс 7 - синий
        8: [0.5, 0.0, 0.5],  # Класс 8 - фиолетовый
    }
    
    colors = np.array([colors_map.get(int(c), [0.5, 0.5, 0.5]) for c in classes])
    
    # Создание 3D графика
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Отрисовка точек
    scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                        c=colors, s=1, alpha=0.6)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Легенда
    if has_classes:
        unique_classes = np.unique(classes)
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor=colors_map.get(int(c), [0.5, 0.5, 0.5]), 
                                     markersize=8, label=f'Class {int(c)}')
                          for c in unique_classes]
        ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.show()
    
    # Статистика
    print("\n📊 Статистика:")
    print(f"Всего точек: {len(las.x)}")
    print(f"Отображено точек: {len(points)}")
    
    if has_classes:
        print("\nРаспределение классов:")
        unique, counts = np.unique(classes, return_counts=True)
        for u, c in zip(unique, counts):
            print(f"  Класс {int(u)}: {c} точек")

def compare_las_files(original_las, predicted_las, max_points=30000):
    """Сравнение оригинала и предсказания"""
    fig = plt.figure(figsize=(20, 8))
    
    # Оригинал
    ax1 = fig.add_subplot(121, projection='3d')
    las1 = laspy.read(original_las)
    points1 = np.vstack([las1.x, las1.y, las1.z]).T
    classes1 = las1.classification
    
    if len(points1) > max_points:
        indices = np.random.choice(len(points1), max_points, replace=False)
        points1 = points1[indices]
        classes1 = classes1[indices]
    
    colors_map = {
        0: [0.5, 0.5, 0.5], 1: [0.6, 0.4, 0.2], 2: [0.3, 0.2, 0.1],
        3: [0.1, 0.5, 0.1], 4: [0.0, 0.3, 0.0], 5: [0.8, 0.8, 0.2],
        6: [0.9, 0.1, 0.1], 7: [0.1, 0.1, 0.9], 8: [0.5, 0.0, 0.5],
    }
    
    colors1 = np.array([colors_map.get(int(c), [0.5, 0.5, 0.5]) for c in classes1])
    ax1.scatter(points1[:, 0], points1[:, 1], points1[:, 2], c=colors1, s=1, alpha=0.6)
    ax1.set_title('Original (Ground Truth)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # Предсказание
    ax2 = fig.add_subplot(122, projection='3d')
    las2 = laspy.read(predicted_las)
    points2 = np.vstack([las2.x, las2.y, las2.z]).T
    classes2 = las2.classification
    
    if len(points2) > max_points:
        indices = np.random.choice(len(points2), max_points, replace=False)
        points2 = points2[indices]
        classes2 = classes2[indices]
    
    colors2 = np.array([colors_map.get(int(c), [0.5, 0.5, 0.5]) for c in classes2])
    ax2.scatter(points2[:, 0], points2[:, 1], points2[:, 2], c=colors2, s=1, alpha=0.6)
    ax2.set_title('Predicted')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        visualize_las(sys.argv[1])
    else:
        print("Использование: python visualize.py <файл.las>")
        print("\nПример:")
        print("  python visualize.py Univer2019.las")
        print("  python visualize.py predicted.las")