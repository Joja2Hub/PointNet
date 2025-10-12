import os
import sys

def check_system():
    """Быстрая проверка системы"""
    issues = []
    
    if not os.path.exists('Univer2019.las'):
        issues.append("❌ Файл Univer2019.las не найден")
    
    if not os.path.exists('model.py'):
        issues.append("❌ Файл model.py не найден")
    
    if not os.path.exists('dataset.py'):
        issues.append("❌ Файл dataset.py не найден")
    
    try:
        import torch
        if not torch.cuda.is_available():
            issues.append("⚠️  CUDA недоступна (обучение будет медленным)")
    except:
        issues.append("❌ PyTorch не установлен")
    
    return issues

def print_menu():
    print("\n" + "="*60)
    print("  POINTNET++ SEMANTIC SEGMENTATION")
    print("="*60)
    print("\n🎯 Выберите действие:")
    print("  1. 🏋️  Обучить модель")
    print("  2. 🔮 Применить модель к unlabeled.las")
    print("  3. 📊 Визуализировать Univer2019.las (размеченный)")
    print("  4. 📊 Визуализировать predicted.las")
    print("  5. 🔄 Сравнить оригинал и предсказание")
    print("  6. 📋 Просмотреть логи")
    print("  7. 🔍 Запустить системный тест")
    print("  8. 🐛 Отладка датасета")
    print("  9. ❌ Выход")
    print("="*60)

def main():
    # Проверка системы при запуске
    print("🔍 Проверка системы...")
    issues = check_system()
    
    if issues:
        print("\n⚠️  Обнаружены проблемы:")
        for issue in issues:
            print(f"   {issue}")
        
        if any("❌" in issue for issue in issues):
            response = input("\nПродолжить? (y/n): ").strip().lower()
            if response != 'y':
                sys.exit(0)
    else:
        print("✅ Система готова к работе!\n")
    
    while True:
        print_menu()
        choice = input("\n👉 Ваш выбор (1-9): ").strip()
        
        if choice == '1':
            print("\n🏋️  Запуск обучения...\n")
            os.system('python train.py')
        
        elif choice == '2':
            if not os.path.exists('checkpoints/best_model.pth'):
                print("\n❌ Модель не найдена! Сначала обучите модель (пункт 1)")
                continue
            
            if not os.path.exists('unlabeled.las'):
                print("\n❌ Файл unlabeled.las не найден!")
                print("💡 Создайте его: python create_unlabeled.py")
                continue
            
            print("\n🔮 Применение модели...\n")
            os.system('python predict.py')
        
        elif choice == '3':
            if not os.path.exists('Univer2019.las'):
                print("\n❌ Файл Univer2019.las не найден!")
                continue
            
            print("\n📊 Визуализация...\n")
            os.system('python visualize.py Univer2019.las')
        
        elif choice == '4':
            if not os.path.exists('predicted.las'):
                print("\n❌ Файл predicted.las не найден! Сначала примените модель (пункт 2)")
                continue
            
            print("\n📊 Визуализация предсказаний...\n")
            os.system('python visualize.py predicted.las')
        
        elif choice == '5':
            if not os.path.exists('Univer2019.las'):
                print("\n❌ Файл Univer2019.las не найден!")
                continue
            
            if not os.path.exists('predicted.las'):
                print("\n❌ Файл predicted.las не найден! Сначала примените модель (пункт 2)")
                continue
            
            print("\n🔄 Сравнение оригинала и предсказания...\n")
            from visualize import compare_las_files
            compare_las_files('Univer2019.las', 'predicted.las')
        
        elif choice == '6':
            print("\n📋 Просмотр логов...\n")
            os.system('python view_logs.py')
        
        elif choice == '7':
            print("\n🔍 Запуск системного теста...\n")
            os.system('python simple_test.py')
        
        elif choice == '8':
            print("\n🐛 Отладка датасета...\n")
            os.system('python debug_dataset.py')
        
        elif choice == '9':
            print("\n👋 До свидания!")
            sys.exit(0)
        
        else:
            print("\n❌ Неверный выбор! Попробуйте снова.")

if __name__ == '__main__':
    main()