import torch
import laspy
import numpy as np
import sys
import platform
from collections import Counter

def diagnose_environment():
    """Диагностика окружения"""
    print("="*60)
    print("ДИАГНОСТИКА ОКРУЖЕНИЯ")
    print("="*60)
    
    # PyTorch и CUDA
    print(f"PyTorch версия: {torch.__version__}")
    print(f"Python версия: {sys.version}")
    print(f"Платформа: {platform.platform()}")
    print(f"Архитектура: {platform.architecture()}")
    
    print(f"\nCUDA доступен: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Количество GPU: {torch.cuda.device_count()}")
        print(f"Текущий GPU: {torch.cuda.current_device()}")
        print(f"Имя GPU: {torch.cuda.get_device_name()}")
        print(f"CUDA версия: {torch.version.cuda}")
    else:
        print("CUDA недоступна - будет использоваться CPU")
    
    # Память
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"Всего GPU памяти: {gpu_memory:.2f} GB")
    
    cpu_memory = torch.get_num_threads()
    print(f"Количество CPU потоков: {cpu_memory}")

def analyze_las_dataset(las_file):
    """Анализ LAS датасета"""
    print(f"\n{'='*60}")
    print(f"АНАЛИЗ LAS ДАТАСЕТА: {las_file}")
    print("="*60)
    
    try:
        las = laspy.read(las_file)
        
        print(f"Формат файла: {las.header.version}")
        print(f"Количество точек: {len(las.points)}")
        print(f"Количество атрибутов: {len(las.point_format.dimensions)}")
        
        # Атрибуты
        print(f"\nДоступные атрибуты:")
        for dim in las.point_format.dimensions:
            print(f"  - {dim.name}")
        
        # Координаты
        x_range = (np.min(las.x), np.max(las.x))
        y_range = (np.min(las.y), np.max(las.y))
        z_range = (np.min(las.z), np.max(las.z))
        
        print(f"\nДиапазоны координат:")
        print(f"  X: {x_range[0]:.2f} - {x_range[1]:.2f}")
        print(f"  Y: {y_range[0]:.2f} - {y_range[1]:.2f}")
        print(f"  Z: {z_range[0]:.2f} - {z_range[1]:.2f}")
        
        # Классификация
        if hasattr(las, 'classification'):
            classifications = las.classification
            unique_classes, counts = np.unique(classifications, return_counts=True)
            
            print(f"\nКлассы разметки:")
            for cls, count in zip(unique_classes, counts):
                print(f"  Класс {cls}: {count} точек ({count/len(classifications)*100:.2f}%)")
            
            print(f"\nВсего уникальных классов: {len(unique_classes)}")
            print(f"Диапазон классов: {unique_classes.min()} - {unique_classes.max()}")
        
        # Проверка других возможных атрибутов
        print(f"\nДополнительные атрибуты:")
        if hasattr(las, 'intensity'):
            print(f"  Intensity: {np.min(las.intensity)} - {np.max(las.intensity)}")
        if hasattr(las, 'return_number'):
            unique_returns, return_counts = np.unique(las.return_number, return_counts=True)
            print(f"  Return numbers: {dict(zip(unique_returns, return_counts))}")
        if hasattr(las, 'number_of_returns'):
            unique_returns, return_counts = np.unique(las.number_of_returns, return_counts=True)
            print(f"  Number of returns: {dict(zip(unique_returns, return_counts))}")
        
        # Размер файла
        import os
        file_size = os.path.getsize(las_file) / (1024**2)  # в MB
        print(f"\nРазмер файла: {file_size:.2f} MB")
        
        return las
        
    except Exception as e:
        print(f"Ошибка при чтении LAS файла: {e}")
        return None

def test_data_loading(las_file, sample_size=10000):
    """Тест загрузки данных"""
    print(f"\n{'='*60}")
    print(f"ТЕСТ ЗАГРУЗКИ ДАННЫХ (первые {sample_size} точек)")
    print("="*60)
    
    try:
        las = laspy.read(las_file)
        
        # Берем сэмпл
        if len(las.points) > sample_size:
            indices = np.random.choice(len(las.points), sample_size, replace=False)
            points = np.vstack((las.x[indices], las.y[indices], las.z[indices])).T
            if hasattr(las, 'classification'):
                labels = las.classification[indices]
        else:
            points = np.vstack((las.x, las.y, las.z)).T
            if hasattr(las, 'classification'):
                labels = las.classification
        
        print(f"Загружено точек: {len(points)}")
        print(f"Форма точек: {points.shape}")
        
        # Нормализация для проверки
        centroid = np.mean(points, axis=0)
        points_centered = points - centroid
        max_distance = np.max(np.sqrt(np.sum(points_centered ** 2, axis=1)))
        
        print(f"Центроид: [{centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f}]")
        print(f"Максимальное расстояние от центра: {max_distance:.2f}")
        
        if 'labels' in locals():
            unique_labels, counts = np.unique(labels, return_counts=True)
            print(f"Уникальные метки в сэмпле: {dict(zip(unique_labels, counts))}")
        
        # Тест преобразования в тензор
        points_tensor = torch.FloatTensor(points).T
        print(f"Успешно преобразовано в тензор PyTorch: {points_tensor.shape}")
        
        return True
        
    except Exception as e:
        print(f"Ошибка при тесте загрузки: {e}")
        return False

def suggest_training_params(las_file):
    """Предложение параметров для обучения"""
    print(f"\n{'='*60}")
    print("РЕКОМЕНДАЦИИ ДЛЯ ОБУЧЕНИЯ")
    print("="*60)
    
    try:
        las = laspy.read(las_file)
        total_points = len(las.points)
        
        # Определение количества классов
        if hasattr(las, 'classification'):
            unique_classes = np.unique(las.classification)
            num_classes = len(unique_classes)
            min_class = unique_classes.min()
            max_class = unique_classes.max()
        else:
            num_classes = 1
            min_class = 0
            max_class = 0
        
        print(f"Общее количество точек: {total_points:,}")
        print(f"Количество классов: {num_classes}")
        print(f"Диапазон классов: {min_class} - {max_class}")
        
        # Рекомендуемые параметры
        if total_points < 100000:
            num_points = 2048
            batch_size = 4
        elif total_points < 1000000:
            num_points = 4096
            batch_size = 2
        else:
            num_points = 4096
            batch_size = 1
        
        print(f"\nРекомендуемые параметры:")
        print(f"  - Размер блока (num_points): {num_points}")
        print(f"  - Размер батча: {batch_size}")
        print(f"  - Количество классов для модели: {num_classes}")
        
        # Оценка памяти
        estimated_memory = (num_points * 3 * 4 * batch_size) / (1024**2)  # в MB
        print(f"  - Оценка памяти на один батч: ~{estimated_memory:.2f} MB")
        
        if torch.cuda.is_available():
            print(f"  - Использовать GPU: Да")
        else:
            print(f"  - Использовать CPU: Да (CUDA недоступна)")
        
        return num_points, batch_size, num_classes
        
    except Exception as e:
        print(f"Ошибка при анализе параметров: {e}")
        return 4096, 2, 8

def main():
    las_file = "Univer2019.las"
    
    # Диагностика окружения
    diagnose_environment()
    
    # Анализ датасета
    las_data = analyze_las_dataset(las_file)
    
    if las_data is not None:
        # Тест загрузки
        test_success = test_data_loading(las_file)
        
        if test_success:
            # Рекомендации
            num_points, batch_size, num_classes = suggest_training_params(las_file)
            
            print(f"\n{'='*60}")
            print("СВОДКА")
            print("="*60)
            print(f"Файл: {las_file}")
            print(f"CUDA доступна: {torch.cuda.is_available()}")
            print(f"Рекомендуемый размер блока: {num_points}")
            print(f"Рекомендуемый размер батча: {batch_size}")
            print(f"Количество классов: {num_classes}")
            
            if not torch.cuda.is_available():
                print(f"\n⚠️  ВНИМАНИЕ: CUDA недоступна!")
                print(f"   Обучение будет происходить на CPU, что может быть медленнее.")
                print(f"   Рассмотрите установку PyTorch с поддержкой CUDA.")
        else:
            print(f"\n❌ Тест загрузки не удался!")
    else:
        print(f"\n❌ Не удалось загрузить LAS файл: {las_file}")

if __name__ == "__main__":
    main()