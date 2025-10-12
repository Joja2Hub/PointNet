import torch
import laspy
import numpy as np
import sys
import platform
import os
import traceback
from datetime import datetime
from collections import Counter

class Logger:
    """Класс для логирования в файл и консоль одновременно"""
    def __init__(self, filename):
        os.makedirs('logs', exist_ok=True)
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()

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
        print("⚠️  CUDA недоступна - будет использоваться CPU")
    
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
        file_size = os.path.getsize(las_file) / (1024**2)  # в MB
        print(f"\nРазмер файла: {file_size:.2f} MB")
        
        return las
        
    except Exception as e:
        print(f"❌ Ошибка при чтении LAS файла: {e}")
        print("\nПолная трассировка ошибки:")
        print(traceback.format_exc())
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
        
        print(f"✅ Загружено точек: {len(points)}")
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
        print(f"✅ Успешно преобразовано в тензор PyTorch: {points_tensor.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при тесте загрузки: {e}")
        print("\nПолная трассировка ошибки:")
        print(traceback.format_exc())
        return False

def test_model_creation():
    """Тест создания модели"""
    print(f"\n{'='*60}")
    print("ТЕСТ СОЗДАНИЯ МОДЕЛИ")
    print("="*60)
    
    try:
        from model import PointNet2SemSeg
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Устройство: {device}")
        
        model = PointNet2SemSeg(num_classes=8).to(device)
        print(f"✅ Модель создана успешно")
        
        # Тест forward pass
        test_input = torch.randn(2, 3, 4096).to(device)
        with torch.no_grad():
            output = model(test_input)
        
        print(f"✅ Forward pass: input {test_input.shape} -> output {output.shape}")
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"📊 Всего параметров: {total_params:,}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при создании модели: {e}")
        print("\nПолная трассировка ошибки:")
        print(traceback.format_exc())
        return False

def test_dataset_creation():
    """Тест создания датасета"""
    print(f"\n{'='*60}")
    print("ТЕСТ СОЗДАНИЯ ДАТАСЕТА")
    print("="*60)
    
    try:
        from dataset import LASDataset
        
        print("Создание датасета...")
        dataset = LASDataset(
            'Univer2019.las',
            num_points=4096,
            block_size=50.0,
            stride=25.0,
            train=True
        )
        
        print(f"✅ Датасет создан: {len(dataset)} блоков")
        
        if len(dataset) == 0:
            print("⚠️  Датасет пустой!")
            return False
        
        # Тест загрузки одного блока
        print("\nТест загрузки блока...")
        points, labels = dataset[0]
        print(f"✅ Блок загружен:")
        print(f"   Points shape: {points.shape}")
        print(f"   Labels shape: {labels.shape}")
        print(f"   Unique labels: {torch.unique(labels).tolist()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при создании датасета: {e}")
        print("\nПолная трассировка ошибки:")
        print(traceback.format_exc())
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
        print(f"❌ Ошибка при анализе параметров: {e}")
        print(traceback.format_exc())
        return 4096, 2, 8

def check_all_files():
    """Проверка наличия всех необходимых файлов"""
    print(f"\n{'='*60}")
    print("ПРОВЕРКА ФАЙЛОВ")
    print("="*60)
    
    required_files = {
        'Univer2019.las': 'Исходный датасет (размеченный)',
        'model.py': 'Архитектура модели',
        'dataset.py': 'Загрузчик данных',
        'train.py': 'Скрипт обучения',
        'predict.py': 'Скрипт предсказания',
        'visualize.py': 'Визуализация',
    }
    
    optional_files = {
        'unlabeled.las': 'Неразмеченный датасет',
        'checkpoints/best_model.pth': 'Обученная модель',
        'predicted.las': 'Результат предсказания',
    }
    
    all_ok = True
    
    print("\nОбязательные файлы:")
    for file, desc in required_files.items():
        if os.path.exists(file):
            size = os.path.getsize(file) / (1024 * 1024)
            print(f"  ✅ {file:20s} ({size:>8.2f} MB) - {desc}")
        else:
            print(f"  ❌ {file:20s} НЕ НАЙДЕН - {desc}")
            all_ok = False
    
    print("\nДополнительные файлы:")
    for file, desc in optional_files.items():
        if os.path.exists(file):
            size = os.path.getsize(file) / (1024 * 1024)
            print(f"  ✅ {file:30s} ({size:>8.2f} MB) - {desc}")
        else:
            print(f"  ⚠️  {file:30s} не найден - {desc}")
    
    return all_ok

def main():
    """Главная функция с полной диагностикой"""
    # Настройка логирования
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f'logs/diagnostic_{timestamp}.log'
    logger = Logger(log_file)
    sys.stdout = logger
    sys.stderr = logger
    
    print(f"📝 Диагностика запущена: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📋 Лог сохраняется в: {log_file}\n")
    
    las_file = "Univer2019.las"
    
    results = {}
    
    # 1. Диагностика окружения
    try:
        diagnose_environment()
        results['Окружение'] = '✅'
    except Exception as e:
        print(f"\n❌ Ошибка в диагностике окружения:")
        print(traceback.format_exc())
        results['Окружение'] = '❌'
    
    # 2. Проверка файлов
    try:
        files_ok = check_all_files()
        results['Файлы'] = '✅' if files_ok else '⚠️'
    except Exception as e:
        print(f"\n❌ Ошибка при проверке файлов:")
        print(traceback.format_exc())
        results['Файлы'] = '❌'
    
    # 3. Анализ датасета
    try:
        las_data = analyze_las_dataset(las_file)
        results['Датасет'] = '✅' if las_data is not None else '❌'
    except Exception as e:
        print(f"\n❌ Ошибка при анализе датасета:")
        print(traceback.format_exc())
        results['Датасет'] = '❌'
        las_data = None
    
    # 4. Тест загрузки данных
    if las_data is not None:
        try:
            test_success = test_data_loading(las_file)
            results['Загрузка данных'] = '✅' if test_success else '❌'
        except Exception as e:
            print(f"\n❌ Ошибка при тесте загрузки:")
            print(traceback.format_exc())
            results['Загрузка данных'] = '❌'
            test_success = False
    else:
        results['Загрузка данных'] = '⏭️'
        test_success = False
    
    # 5. Тест создания модели
    try:
        model_ok = test_model_creation()
        results['Модель'] = '✅' if model_ok else '❌'
    except Exception as e:
        print(f"\n❌ Ошибка при тесте модели:")
        print(traceback.format_exc())
        results['Модель'] = '❌'
    
    # 6. Тест создания датасета
    try:
        dataset_ok = test_dataset_creation()
        results['Dataset класс'] = '✅' if dataset_ok else '❌'
    except Exception as e:
        print(f"\n❌ Ошибка при тесте Dataset:")
        print(traceback.format_exc())
        results['Dataset класс'] = '❌'
    
    # 7. Рекомендации
    if test_success:
        try:
            num_points, batch_size, num_classes = suggest_training_params(las_file)
            results['Рекомендации'] = '✅'
        except Exception as e:
            print(f"\n❌ Ошибка при формировании рекомендаций:")
            print(traceback.format_exc())
            results['Рекомендации'] = '❌'
    else:
        results['Рекомендации'] = '⏭️'
    
    # Итоговая сводка
    print(f"\n{'='*60}")
    print("ИТОГОВАЯ СВОДКА")
    print("="*60)
    
    for test_name, status in results.items():
        print(f"  {test_name:25s} {status}")
    
    # Подсчет статуса
    passed = sum(1 for s in results.values() if s == '✅')
    failed = sum(1 for s in results.values() if s == '❌')
    warnings = sum(1 for s in results.values() if s == '⚠️')
    skipped = sum(1 for s in results.values() if s == '⏭️')
    
    print(f"\n📊 Статистика:")
    print(f"  Пройдено: {passed}")
    print(f"  Ошибок: {failed}")
    print(f"  Предупреждений: {warnings}")
    print(f"  Пропущено: {skipped}")
    
    print(f"\n{'='*60}")
    
    if failed == 0 and warnings == 0:
        print("✅ ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ!")
        print("="*60)
        print("\n💡 Система готова к обучению!")
        print("\nСледующие шаги:")
        print("  1. python train.py - начать обучение")
        print("  2. python quick_start.py - автоматический запуск")
    elif failed > 0:
        print("❌ ОБНАРУЖЕНЫ КРИТИЧЕСКИЕ ОШИБКИ!")
        print("="*60)
        print(f"\n📋 Полный лог ошибок сохранен в: {log_file}")
        print("\n💡 Рекомендации:")
        if 'Модель' in results and results['Модель'] == '❌':
            print("  • Проверьте файл model.py")
        if 'Dataset класс' in results and results['Dataset класс'] == '❌':
            print("  • Проверьте файл dataset.py")
        if 'Датасет' in results and results['Датасет'] == '❌':
            print("  • Проверьте файл Univer2019.las")
    else:
        print("⚠️  ЕСТЬ ПРЕДУПРЕЖДЕНИЯ")
        print("="*60)
        print(f"\n📋 Лог сохранен в: {log_file}")
        print("\n💡 Можно попробовать запустить обучение, но могут быть проблемы")
    
    # Финальная информация
    print(f"\n📝 Полный отчет сохранен в: {log_file}")
    print(f"⏰ Время завершения: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Закрываем лог
    logger.close()
    sys.stdout = logger.terminal
    sys.stderr = logger.terminal
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)