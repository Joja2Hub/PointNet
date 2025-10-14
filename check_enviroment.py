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
    """Расширенная диагностика окружения"""
    print("="*60)
    print("ДИАГНОСТИКА ОКРУЖЕНИЯ")
    print("="*60)
    
    # Информация о системе
    memory_gb, disk_free = get_system_info()
    
    print(f"PyTorch версия: {torch.__version__}")
    print(f"Python версия: {sys.version}")
    print(f"Платформа: {platform.platform()}")
    print(f"Архитектура: {platform.architecture()}")
    
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
        print(f"CUDA версия: {torch.version.cuda}")
        
        # Проверка производительности CUDA
        try:
            with torch.no_grad():
                a = torch.randn(1000, 1000).cuda()
                b = torch.randn(1000, 1000).cuda()
                torch.matmul(a, b)
            print("✅ CUDA операции работают корректно")
        except Exception as e:
            issues.append(f"❌ Ошибка в CUDA операциях: {e}")
    else:
        print("ℹ️  Будет использоваться CPU")
    
    cpu_threads = torch.get_num_threads()
    print(f"Количество CPU потоков: {cpu_threads}")
    print(f"Доступно CPU ядер: {os.cpu_count()}")

def check_file_structure(issues):
    """Расширенная проверка файловой структуры"""
    print(f"\n{'='*60}")
    print("ПРОВЕРКА ФАЙЛОВОЙ СТРУКТУРЫ")
    print("="*60)
    
    required_files = {
        'model.py': 'Архитектура модели',
        'dataset.py': 'Загрузчик данных', 
        'train.py': 'Скрипт обучения',
        'predict.py': 'Скрипт предсказания',
        'visualize.py': 'Визуализация результатов',
    }
    
    optional_files = {
        'utils.py': 'Вспомогательные функции',
        'config.py': 'Конфигурация',
        'requirements.txt': 'Зависимости',
    }
    
    required_dirs = {
        'datasets/row': 'Размеченные данные',
        'datasets/unlabeled': 'Неразмеченные данные',
        'logs': 'Логи и диагностика',
        'checkpoints': 'Сохраненные модели',
    }
    
    all_ok = True
    
    print("\n📁 ОБЯЗАТЕЛЬНЫЕ ПАПКИ:")
    for dir_path, desc in required_dirs.items():
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            files_count = len([f for f in os.listdir(dir_path) if not f.startswith('.')])
            las_files = [f for f in os.listdir(dir_path) if f.lower().endswith(('.las', '.laz'))]
            las_count = len(las_files)
            
            status_icon = "✅" if files_count > 0 else "⚠️"
            print(f"  {status_icon} {dir_path:25s} ({files_count} файлов, {las_count} LAS) - {desc}")
            
            if las_count > 0 and 'dataset' in dir_path:
                # Показываем примеры LAS файлов
                sample_files = las_files[:3]
                if len(las_files) > 3:
                    sample_files.append(f"... всего {las_count} файлов")
                print(f"        📄 {', '.join(sample_files)}")
        else:
            print(f"  ❌ {dir_path:25s} НЕ НАЙДЕНА - {desc}")
            issues.append(f"❌ Отсутствует обязательная папка: {dir_path}")
            all_ok = False
    
    print("\n📄 ОБЯЗАТЕЛЬНЫЕ ФАЙЛЫ:")
    for file, desc in required_files.items():
        if os.path.exists(file):
            size_mb = os.path.getsize(file) / (1024 * 1024)
            with open(file, 'r', encoding='utf-8') as f:
                lines = len(f.readlines())
            print(f"  ✅ {file:20s} ({size_mb:6.2f} MB, {lines:4} строк) - {desc}")
        else:
            print(f"  ❌ {file:20s} НЕ НАЙДЕН - {desc}")
            issues.append(f"❌ Отсутствует обязательный файл: {file}")
            all_ok = False
    
    print("\n📄 ДОПОЛНИТЕЛЬНЫЕ ФАЙЛЫ:")
    for file, desc in optional_files.items():
        if os.path.exists(file):
            size_mb = os.path.getsize(file) / (1024 * 1024)
            print(f"  ✅ {file:20s} ({size_mb:6.2f} MB) - {desc}")
        else:
            print(f"  ⚠️  {file:20s} не найден - {desc}")
    
    return all_ok

def analyze_model_complexity(model, device):
    """Анализ сложности модели"""
    total_params = 0
    layer_info = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_count = param.numel()
            total_params += param_count
            layer_info.append((name, param_count, param.shape))
    
    # Сортировка по количеству параметров
    layer_info.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n📊 АНАЛИЗ СЛОЖНОСТИ МОДЕЛИ:")
    print(f"Всего обучаемых параметров: {total_params:,}")
    print(f"Размер модели в памяти: ~{total_params * 4 / (1024**2):.2f} MB (float32)")
    
    # Топ-5 самых больших слоев
    print("\nТоп-5 самых больших слоев:")
    for name, count, shape in layer_info[:5]:
        size_mb = count * 4 / (1024**2)
        print(f"  {name:40s} {count:>9,} params ({size_mb:5.2f} MB) {str(shape):>20s}")
    
    return total_params

def test_model_creation(issues):
    """Расширенный тест создания модели"""
    print(f"\n{'='*60}")
    print("ТЕСТ СОЗДАНИЯ И ПРОИЗВОДИТЕЛЬНОСТИ МОДЕЛИ")
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
        
        # Тест с разным количеством классов
        test_configs = [4, 8, 12]
        model_ok = True
        
        for num_classes in test_configs:
            try:
                print(f"\n🧪 Тест с {num_classes} классами:")
                model = PointNet2SemSeg(num_classes=num_classes).to(device)
                model.eval()
                
                # Тест разных размеров батча
                batch_sizes = [1, 2, 4]
                for batch_size in batch_sizes:
                    try:
                        # Тест forward pass
                        test_input = torch.randn(batch_size, 3, 4096).to(device)
                        
                        # Замер памяти до inference
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            memory_before = torch.cuda.memory_allocated()
                        
                        # Замер времени inference
                        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                        end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                        
                        if start_time:
                            start_time.record()
                        
                        with torch.no_grad():
                            output = model(test_input)
                        
                        if end_time:
                            end_time.record()
                            torch.cuda.synchronize()
                            inference_time = start_time.elapsed_time(end_time)
                        
                        # Замер памяти после inference
                        if torch.cuda.is_available():
                            memory_after = torch.cuda.memory_allocated()
                            memory_used = (memory_after - memory_before) / (1024**2)
                        
                        print(f"  ✅ Батч {batch_size}: input {test_input.shape} -> output {output.shape}")
                        
                        if torch.cuda.is_available():
                            print(f"      ⏱️  Время: {inference_time:.1f} ms, 💾 Память: {memory_used:.1f} MB")
                        
                        # Проверка выходных значений
                        if torch.isnan(output).any():
                            issues.append(f"⚠️  Обнаружены NaN в выходных данных (batch_size={batch_size})")
                        if torch.isinf(output).any():
                            issues.append(f"⚠️  Обнаружены Inf в выходных данных (batch_size={batch_size})")
                            
                        del test_input, output
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            
                    except Exception as e:
                        print(f"  ❌ Ошибка с batch_size={batch_size}: {e}")
                        model_ok = False
                
                # Анализ сложности модели
                if num_classes == 8:  # Только для стандартной конфигурации
                    total_params = analyze_model_complexity(model, device)
                
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"❌ Ошибка при создании модели с {num_classes} классами: {e}")
                model_ok = False
        
        if model_ok:
            print(f"\n🎯 РЕКОМЕНДАЦИИ ПО МОДЕЛИ:")
            print(f"  • Поддерживается до 12 классов")
            print(f"  • Рекомендуемый размер батча: 2-4")
            if torch.cuda.is_available():
                print(f"  • GPU память достаточна для инференса")
            else:
                print(f"  • Используется CPU - обучение может быть медленным")
        
        return model_ok
        
    except Exception as e:
        error_msg = f"❌ Критическая ошибка при тесте модели: {e}"
        print(error_msg)
        print("\nПолная трассировка ошибки:")
        print(traceback.format_exc())
        issues.append(error_msg)
        return False

def test_dataset_functionality(issues):
    """Расширенный тест функциональности датасета"""
    print(f"\n{'='*60}")
    print("ТЕСТ ФУНКЦИОНАЛЬНОСТИ ДАТАСЕТА")
    print("="*60)
    
    try:
        from dataset import LASDataset
        
        # Поиск LAS файлов
        las_files = []
        if os.path.exists('datasets/row'):
            las_files = [f for f in os.listdir('datasets/row') if f.lower().endswith(('.las', '.laz'))]
            las_files = [os.path.join('datasets', 'row', f) for f in las_files]
        
        if not las_files:
            msg = "⚠️  Не найдено LAS-файлов в datasets/row/"
            print(msg)
            issues.append(msg)
            return False
        
        test_file = las_files[0]
        print(f"Используется файл: {test_file}")
        
        # Тест разных конфигураций датасета
        configs = [
            {"num_points": 4096, "block_size": 50.0, "stride": 25.0},
            {"num_points": 2048, "block_size": 30.0, "stride": 15.0},
            {"num_points": 8192, "block_size": 100.0, "stride": 50.0},
        ]
        
        dataset_ok = True
        
        for i, config in enumerate(configs):
            try:
                print(f"\n🧪 Конфигурация {i+1}: {config}")
                dataset = LASDataset(
                    test_file,
                    num_points=config["num_points"],
                    block_size=config["block_size"],
                    stride=config["stride"],
                    train=True
                )
                
                print(f"  ✅ Создано блоков: {len(dataset)}")
                
                if len(dataset) > 0:
                    # Тест загрузки нескольких блоков
                    for j in range(min(3, len(dataset))):
                        points, labels = dataset[j]
                        print(f"  📦 Блок {j}: points {points.shape}, labels {labels.shape}")
                        
                        # Проверка данных
                        if torch.isnan(points).any():
                            issues.append(f"⚠️  Обнаружены NaN в точках (конфиг {i+1}, блок {j})")
                        if torch.isinf(points).any():
                            issues.append(f"⚠️  Обнаружены Inf в точках (конфиг {i+1}, блок {j})")
                        
                        unique_labels = torch.unique(labels)
                        print(f"      Метки: {unique_labels.tolist()}")
                
                # Тест трансформаций
                try:
                    from dataset import train_transforms, test_transforms
                    print("  ✅ Трансформации загружены успешно")
                except ImportError:
                    print("  ⚠️  Трансформации не найдены в dataset.py")
                
                del dataset
                
            except Exception as e:
                print(f"  ❌ Ошибка в конфигурации {i+1}: {e}")
                dataset_ok = False
        
        if dataset_ok:
            print(f"\n🎯 РЕКОМЕНДАЦИИ ПО ДАТАСЕТУ:")
            print(f"  • Используйте num_points=4096 для баланса скорости и качества")
            print(f"  • Block_size=50.0 и stride=25.0 дают хорошее перекрытие")
            print(f"  • Доступно {len(las_files)} LAS файлов")
        
        return dataset_ok
        
    except Exception as e:
        error_msg = f"❌ Ошибка при тесте датасета: {e}"
        print(error_msg)
        print("\nПолная трассировка ошибки:")
        print(traceback.format_exc())
        issues.append(error_msg)
        return False

def test_training_components(issues):
    """Тест компонентов обучения"""
    print(f"\n{'='*60}")
    print("ТЕСТ КОМПОНЕНТОВ ОБУЧЕНИЯ")
    print("="*60)
    
    components_ok = True
    
    # Проверка наличия основных компонентов обучения
    try:
        import torch.nn as nn
        import torch.optim as optim
        
        # Тест функции потерь
        loss_functions = {
            'CrossEntropy': nn.CrossEntropyLoss(),
            'NLLLoss': nn.NLLLoss(),
        }
        
        print("📊 Функции потерь:")
        for name, loss_fn in loss_functions.items():
            try:
                # Тестовый forward pass
                outputs = torch.randn(2, 8, 4096)
                targets = torch.randint(0, 8, (2, 4096))
                loss = loss_fn(outputs, targets)
                print(f"  ✅ {name:15s} - loss: {loss.item():.4f}")
            except Exception as e:
                print(f"  ❌ {name:15s} - ошибка: {e}")
                components_ok = False
        
        # Тест оптимизаторов
        print("\n⚡ Оптимизаторы:")
        try:
            from model import PointNet2SemSeg
            model = PointNet2SemSeg(num_classes=8)
            
            optimizers = {
                'Adam': optim.Adam(model.parameters(), lr=0.001),
                'SGD': optim.SGD(model.parameters(), lr=0.01),
                'AdamW': optim.AdamW(model.parameters(), lr=0.001),
            }
            
            for name, optimizer in optimizers.items():
                print(f"  ✅ {name:15s} - инициализирован")
                
            del model
                
        except Exception as e:
            print(f"  ❌ Ошибка теста оптимизаторов: {e}")
            components_ok = False
        
        # Проверка скрипта обучения
        print("\n📜 Скрипт обучения:")
        if os.path.exists('train.py'):
            with open('train.py', 'r', encoding='utf-8') as f:
                content = f.read()
                has_training_loop = 'for epoch' in content and 'optimizer.zero_grad()' in content
                has_validation = 'model.eval()' in content and 'with torch.no_grad()' in content
                has_checkpoints = 'torch.save' in content
                
                checks = [
                    ("Цикл обучения", has_training_loop),
                    ("Валидация", has_validation),
                    ("Сохранение чекпоинтов", has_checkpoints),
                ]
                
                for check_name, check_result in checks:
                    icon = "✅" if check_result else "⚠️"
                    print(f"  {icon} {check_name}")
                    
                if not all(check for _, check in checks):
                    issues.append("⚠️  В train.py отсутствуют некоторые ключевые компоненты")
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
    """Главная функция с полной диагностикой"""
    issues = []
    
    # Настройка логирования
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f'logs/diagnostic_{timestamp}.log'
    logger = Logger(log_file)
    sys.stdout = logger
    sys.stderr = logger
    
    print(f"📝 Диагностика запущена: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📋 Лог сохраняется в: {log_file}\n")
    
    results = {}
    
    # Запуск всех тестов
    tests = [
        ("Окружение", lambda: diagnose_environment(issues) or True),
        ("Файловая структура", lambda: check_file_structure(issues)),
        ("Модель", lambda: test_model_creation(issues)),
        ("Датасет", lambda: test_dataset_functionality(issues)),
        ("Компоненты обучения", lambda: test_training_components(issues)),
    ]
    
    for test_name, test_func in tests:
        try:
            print(f"\n{'🚀' * 20} ТЕСТ: {test_name} {'🚀' * 20}\n")
            success = test_func()
            results[test_name] = '✅' if success else '❌'
        except Exception as e:
            error_msg = f"❌ Неожиданная ошибка в тесте {test_name}: {e}"
            print(error_msg)
            print(traceback.format_exc())
            issues.append(error_msg)
            results[test_name] = '❌'
    
    # Итоговая сводка
    print(f"\n{'='*60}")
    print("ИТОГОВАЯ СВОДКА ДИАГНОСТИКИ")
    print("="*60)
    
    for test_name, status in results.items():
        print(f"  {test_name:25s} {status}")
    
    # Статистика
    passed = sum(1 for s in results.values() if s == '✅')
    failed = sum(1 for s in results.values() if s == '❌')
    total = len(results)
    
    print(f"\n📊 Статистика:")
    print(f"  Пройдено: {passed}/{total}")
    print(f"  Ошибок: {failed}")
    print(f"  Успешность: {passed/total*100:.1f}%")
    
    # Вывод проблем и рекомендаций
    if issues:
        print(f"\n{'='*60}")
        print("ОБНАРУЖЕННЫЕ ПРОБЛЕМЫ И РЕКОМЕНДАЦИИ:")
        print("="*60)
        for i, issue in enumerate(issues, 1):
            print(f"  {i:2d}. {issue}")
    
    print(f"\n{'='*60}")
    
    if failed == 0:
        print("✅ ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ!")
        print("\n💡 Система полностью готова к обучению!")
        print("\n🎯 СЛЕДУЮЩИЕ ШАГИ:")
        print("  1. python train.py - начать обучение модели")
        print("  2. Используйте 'Анализ датасета' для детального просмотра данных")
        print("  3. Настройте параметры в config.py (если есть)")
        print("  4. Запустите обучение и мониторьте логи в папке logs/")
    else:
        print("❌ ОБНАРУЖЕНЫ ПРОБЛЕМЫ!")
        print("\n🔧 РЕКОМЕНДАЦИИ ПО РЕШЕНИЮ:")
        if any("модел" in issue.lower() for issue in issues):
            print("  • Проверьте архитектуру модели в model.py")
            print("  • Убедитесь в совместимости версий PyTorch")
        if any("датасет" in issue.lower() for issue in issues):
            print("  • Проверьте наличие LAS файлов в datasets/row/")
            print("  • Убедитесь в корректности dataset.py")
        if any("файл" in issue.lower() for issue in issues):
            print("  • Проверьте наличие всех обязательных файлов")
            print("  • Убедитесь в правильности структуры папок")
    
    # Финальная информация
    print(f"\n📝 Полный отчет сохранен в: {log_file}")
    print(f"⏰ Время завершения: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Очистка памяти
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    # Закрываем лог
    logger.close()
    sys.stdout = logger.terminal
    sys.stderr = logger.terminal
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)