import torch
import numpy as np
import sys
import platform
import os
import traceback
import psutil
import gc
from datetime import datetime

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

def get_system_info():
    """Получение информации о системе"""
    try:
        memory_gb = psutil.virtual_memory().total / (1024**3)
        disk_usage = psutil.disk_usage('.').free / (1024**3)
        return memory_gb, disk_usage
    except:
        return None, None

def diagnose_environment(issues):
    """Оптимистичная диагностика окружения"""
    print("="*60)
    print("ДИАГНОСТИКА ОКРУЖЕНИЯ")
    print("="*60)
    
    # Информация о системе
    memory_gb, disk_free = get_system_info()
    
    print(f"PyTorch версия: {torch.__version__}")
    print(f"Python версия: {sys.version}")
    print(f"Платформа: {platform.platform()}")
    
    if memory_gb:
        print(f"Оперативная память: {memory_gb:.1f} GB")
    if disk_free:
        print(f"Свободное место на диске: {disk_free:.1f} GB")
    
    # Детальная информация о CUDA
    cuda_available = torch.cuda.is_available()
    print(f"\nCUDA доступен: {cuda_available}")
    
    if cuda_available:
        print(f"Количество GPU: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            print(f"  GPU {i}: {props.name} ({memory_gb:.1f} GB)")
        print(f"Текущий GPU: {torch.cuda.current_device()}")
        
        # Быстрая проверка CUDA
        try:
            with torch.no_grad():
                a = torch.randn(100, 100).cuda()
                b = torch.randn(100, 100).cuda()
                torch.matmul(a, b)
            print("✅ CUDA операции работают корректно")
        except Exception as e:
            issues.append(f"❌ Ошибка в CUDA операциях: {e}")
    else:
        print("ℹ️  Будет использоваться CPU")
    
    print(f"Доступно CPU ядер: {os.cpu_count()}")
    return True

def check_file_structure(issues):
    """Упрощенная проверка файловой структуры"""
    print(f"\n{'='*60}")
    print("ПРОВЕРКА ФАЙЛОВОЙ СТРУКТУРЫ")
    print("="*60)
    
    required_files = {
        'model.py': 'Архитектура модели',
        'dataset.py': 'Загрузчик данных', 
        'train.py': 'Скрипт обучения',
    }
    
    optional_files = {
        'predict.py': 'Скрипт предсказания',
        'visualize.py': 'Визуализация результатов',
        'utils.py': 'Вспомогательные функции',
        'config.py': 'Конфигурация',
        'requirements.txt': 'Зависимости',
    }
    
    required_dirs = {
        'datasets': 'Данные',
        'logs': 'Логи и диагностика',
        'checkpoints': 'Сохраненные модели',
    }
    
    all_ok = True
    
    print("\n📁 ОСНОВНЫЕ ПАПКИ:")
    for dir_path, desc in required_dirs.items():
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            files_count = len([f for f in os.listdir(dir_path) if not f.startswith('.')])
            status_icon = "✅" if files_count > 0 else "⚠️"
            print(f"  {status_icon} {dir_path:20s} ({files_count} файлов) - {desc}")
        else:
            print(f"  ❌ {dir_path:20s} НЕ НАЙДЕНА - {desc}")
            issues.append(f"❌ Отсутствует папка: {dir_path}")
            all_ok = False
    
    print("\n📄 ОСНОВНЫЕ ФАЙЛЫ:")
    for file, desc in required_files.items():
        if os.path.exists(file):
            size_kb = os.path.getsize(file) / 1024
            print(f"  ✅ {file:20s} ({size_kb:6.1f} KB) - {desc}")
        else:
            print(f"  ❌ {file:20s} НЕ НАЙДЕН - {desc}")
            issues.append(f"❌ Отсутствует файл: {file}")
            all_ok = False
    
    print("\n📄 ДОПОЛНИТЕЛЬНЫЕ ФАЙЛЫ:")
    for file, desc in optional_files.items():
        if os.path.exists(file):
            size_kb = os.path.getsize(file) / 1024
            print(f"  ✅ {file:20s} ({size_kb:6.1f} KB) - {desc}")
        else:
            print(f"  ⚠️  {file:20s} не найден - {desc}")
    
    return all_ok

def analyze_model_complexity(model, device):
    """Упрощенный анализ сложности модели"""
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n📊 АНАЛИЗ СЛОЖНОСТИ МОДЕЛИ:")
    print(f"Всего обучаемых параметров: {total_params:,}")
    print(f"Размер модели в памяти: ~{total_params * 4 / (1024**2):.2f} MB (float32)")
    
    return total_params

def test_model_creation(issues):
    """Оптимистичный тест создания модели"""
    print(f"\n{'='*60}")
    print("ТЕСТ СОЗДАНИЯ МОДЕЛИ")
    print("="*60)
    
    try:
        # Попытка импорта модели
        try:
            from model import PointNet2SemSeg
            print("✅ Модуль model.py загружен успешно")
        except ImportError as e:
            issues.append(f"❌ Ошибка импорта из model.py: {e}")
            return False
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Устройство: {device}")
        
        # Тест с стандартным количеством классов
        try:
            print(f"\n🧪 Тест с 8 классами:")
            model = PointNet2SemSeg(num_classes=8).to(device)
            model.eval()
            
            # Быстрый тест forward pass
            test_input = torch.randn(2, 3, 4096).to(device)
            
            with torch.no_grad():
                output = model(test_input)
            
            print(f"  ✅ Батч 2: input {test_input.shape} -> output {output.shape}")
            
            # Анализ сложности модели
            total_params = analyze_model_complexity(model, device)
            
            del model, test_input, output
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            print(f"\n🎯 МОДЕЛЬ ГОТОВА К РАБОТЕ!")
            print(f"  • Успешно создана с 8 классами")
            print(f"  • Forward pass работает корректно")
            print(f"  • Параметров: {total_params:,}")
            
            return True
            
        except Exception as e:
            print(f"❌ Ошибка при создании модели: {e}")
            issues.append(f"❌ Ошибка создания модели: {e}")
            return False
        
    except Exception as e:
        error_msg = f"❌ Критическая ошибка при тесте модели: {e}"
        print(error_msg)
        issues.append(error_msg)
        return False

def test_dataset_functionality(issues):
    """Улучшенная проверка структуры и наличия датасетов"""
    print(f"\n{'='*60}")
    print("ПРОВЕРКА СТРУКТУРЫ ДАТАСЕТОВ")
    print("="*60)
    
    all_ok = True

    # 1. Проверка импорта модуля dataset.py
    try:
        from dataset import LASDataset
        print("✅ Модуль dataset.py загружен успешно")
    except Exception as e:
        error_msg = f"❌ Ошибка импорта из dataset.py: {e}"
        print(error_msg)
        issues.append(error_msg)
        all_ok = False

    # 2. Проверка трансформаций (опционально)
    try:
        from dataset import train_transforms, test_transforms
        print("✅ Трансформации (train/test) доступны")
    except (ImportError, AttributeError):
        print("⚠️  Трансформации не найдены в dataset.py — могут понадобиться позже")

    # 3. Проверка структуры папок с данными
    raw_dir = os.path.join('datasets', 'raw')
    unlabeled_dir = os.path.join('datasets', 'unlabeled')

    for dir_path, desc in [
        (raw_dir, "сырые данные (.las/.laz)"),
        (unlabeled_dir, "очищенные/предобработанные данные")
    ]:
        print(f"\n📁 Папка: {dir_path} ({desc})")
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f)) and not f.startswith('.')]
            if files:
                # Группировка по расширению
                ext_count = {}
                total_size = 0
                for f in files:
                    ext = os.path.splitext(f)[1].lower()
                    ext_count[ext] = ext_count.get(ext, 0) + 1
                    total_size += os.path.getsize(os.path.join(dir_path, f))
                
                ext_str = ', '.join([f"{count}×{ext}" for ext, count in sorted(ext_count.items())])
                total_size_mb = total_size / (1024**2)
                print(f"  ✅ Найдено файлов: {len(files)} ({ext_str}) | {total_size_mb:.1f} MB")
            else:
                print(f"  ⚠️  Папка существует, но пуста")
                if 'raw' in dir_path:
                    issues.append(f"⚠️  Папка {dir_path} пуста — добавьте .las/.laz файлы")
        else:
            print(f"  ❌ Папка не найдена")
            issues.append(f"❌ Отсутствует папка: {dir_path}")
            all_ok = False

    # 4. Тест инициализации LASDataset (без реального файла)
    try:
        dataset = LASDataset(
            os.path.join('datasets', 'raw', 'dummy.las'),
            num_points=4096,
            block_size=50.0,
            stride=25.0,
            train=True
        )
        print("\n🧪 Класс LASDataset: инициализация возможна")
    except Exception as e:
        if "dummy.las" in str(e) or "No such file" in str(e) or "не найден" in str(e).lower():
            print("\n🧪 Класс LASDataset: инициализация возможна (ошибка файла ожидаема)")
        else:
            error_msg = f"❌ Неожиданная ошибка в LASDataset.__init__: {e}"
            print(error_msg)
            issues.append(error_msg)
            all_ok = False

    return all_ok

def test_training_components(issues):
    """Упрощенный тест компонентов обучения"""
    print(f"\n{'='*60}")
    print("ПРОВЕРКА КОМПОНЕНТОВ ОБУЧЕНИЯ")
    print("="*60)
    
    components_ok = True
    
    try:
        import torch.nn as nn
        import torch.optim as optim
        
        # Быстрая проверка функции потерь
        print("📊 Функции потерь:")
        try:
            loss_fn = nn.CrossEntropyLoss()
            outputs = torch.randn(2, 8, 4096)
            targets = torch.randint(0, 8, (2, 4096))
            loss = loss_fn(outputs, targets)
            print(f"  ✅ CrossEntropyLoss - работает (loss: {loss.item():.4f})")
        except Exception as e:
            print(f"  ❌ CrossEntropyLoss - ошибка: {e}")
            components_ok = False
        
        # Быстрая проверка оптимизаторов
        print("\n⚡ Оптимизаторы:")
        try:
            from model import PointNet2SemSeg
            model = PointNet2SemSeg(num_classes=8)
            
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            print(f"  ✅ Adam - инициализирован")
                
            del model
                
        except Exception as e:
            print(f"  ❌ Ошибка теста оптимизаторов: {e}")
            components_ok = False
        
        # Базовая проверка скрипта обучения
        print("\n📜 Скрипт обучения:")
        if os.path.exists('train.py'):
            with open('train.py', 'r', encoding='utf-8') as f:
                content = f.read()
                has_training = 'def train' in content or 'for epoch' in content
                has_model = 'PointNet2SemSeg' in content or 'model' in content
                
                if has_training and has_model:
                    print("  ✅ train.py содержит основные компоненты")
                else:
                    print("  ⚠️  train.py может быть неполным")
        else:
            print("  ❌ train.py не найден")
            components_ok = False
        
        return components_ok
        
    except Exception as e:
        error_msg = f"❌ Ошибка при тесте компонентов обучения: {e}"
        print(error_msg)
        issues.append(error_msg)
        return False

def main():
    """Главная функция с оптимистичной диагностикой"""
    issues = []
    
    # Настройка логирования
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f'logs/diagnostic_{timestamp}.log'
    logger = Logger(log_file)
    sys.stdout = logger
    sys.stderr = logger
    
    print(f"🚀 ОПТИМИСТИЧНАЯ ДИАГНОСТИКА ЗАПУЩЕНА")
    print(f"📋 Лог сохраняется в: {log_file}\n")
    
    results = {}
    
    # Запуск упрощенных тестов
    tests = [
        ("Окружение", lambda: diagnose_environment(issues)),
        ("Файловая структура", lambda: check_file_structure(issues)),
        ("Модель", lambda: test_model_creation(issues)),
        ("Датасет", lambda: test_dataset_functionality(issues)),
        ("Компоненты обучения", lambda: test_training_components(issues)),
    ]
    
    for test_name, test_func in tests:
        try:
            print(f"\n{'🔍' * 10} ТЕСТ: {test_name} {'🔍' * 10}\n")
            success = test_func()
            results[test_name] = '✅' if success else '❌'
        except Exception as e:
            error_msg = f"❌ Неожиданная ошибка в тесте {test_name}: {e}"
            print(error_msg)
            issues.append(error_msg)
            results[test_name] = '❌'
    
    # Итоговая сводка
    print(f"\n{'='*60}")
    print("ИТОГОВАЯ СВОДКА")
    print("="*60)
    
    for test_name, status in results.items():
        print(f"  {test_name:25s} {status}")
    
    # Статистика
    passed = sum(1 for s in results.values() if s == '✅')
    total = len(results)
    
    print(f"\n📊 Результат: {passed}/{total} пройдено")
    
    # Вывод проблем
    if issues:
        print(f"\n⚠️  Обнаруженные проблемы:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i:2d}. {issue}")
    else:
        print(f"\n✅ Проблем не обнаружено!")
    
    print(f"\n{'='*60}")
    
    if passed == total:
        print("🎉 ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ!")
        print("\n💡 Система готова к работе!")
        print("\n🎯 СЛЕДУЮЩИЕ ШАГИ:")
        print("  1. Добавьте данные в папку datasets/")
        print("  2. Настройте параметры обучения при необходимости")
        print("  3. Запустите: python train.py")
    else:
        print("⚠️  ЕСТЬ ПРОБЛЕМЫ ДЛЯ РЕШЕНИЯ")
        print("\n🔧 Проверьте указанные выше проблемы и повторите диагностику")
    
    # Финальная информация
    print(f"\n📝 Полный отчет: {log_file}")
    
    # Очистка памяти
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    # Закрываем лог
    logger.close()
    sys.stdout = logger.terminal
    sys.stderr = logger.terminal
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)