"""
Быстрый старт: обучение и предсказание одной командой
"""
import os
import sys
from datetime import datetime

def main():
    print("="*60)
    print("  🚀 БЫСТРЫЙ СТАРТ: ОБУЧЕНИЕ И ПРЕДСКАЗАНИЕ")
    print("="*60)
    
    # Создание папки для логов
    os.makedirs('logs', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Проверка файлов
    if not os.path.exists('Univer2019.las'):
        print("\n❌ Ошибка: Файл Univer2019.las не найден!")
        sys.exit(1)
    
    if not os.path.exists('unlabeled.las'):
        print("\n❌ Ошибка: Файл unlabeled.las не найден!")
        print("💡 Создайте его с помощью create_unlabeled.py")
        sys.exit(1)
    
    # Шаг 1: Обучение
    print("\n" + "="*60)
    print("ШАГ 1/2: ОБУЧЕНИЕ МОДЕЛИ")
    print("="*60)
    
    response = input("\n🏋️  Начать обучение? (y/n): ").strip().lower()
    if response != 'y':
        print("❌ Обучение отменено")
        sys.exit(0)
    
    print("\n⏳ Обучение начато (это может занять несколько часов)...\n")
    
    # Логирование обучения
    train_log = f'logs/train_{timestamp}.log'
    print(f"📝 Логи обучения сохраняются в: {train_log}\n")
    
    exit_code = os.system(f'python train.py 2>&1 | tee {train_log}')
    
    if exit_code != 0:
        print(f"\n❌ ОШИБКА при обучении!")
        print(f"📋 Проверьте лог: {train_log}")
        sys.exit(1)
    
    if not os.path.exists('checkpoints/best_model.pth'):
        print(f"\n❌ Модель не была сохранена! Проверьте ошибки обучения.")
        print(f"📋 Проверьте лог: {train_log}")
        sys.exit(1)
    
    print("\n✅ Обучение завершено!")
    
    # Шаг 2: Предсказание
    print("\n" + "="*60)
    print("ШАГ 2/2: ПРИМЕНЕНИЕ МОДЕЛИ")
    print("="*60)
    
    response = input("\n🔮 Применить модель к unlabeled.las? (y/n): ").strip().lower()
    if response != 'y':
        print("⏭️  Предсказание пропущено")
        sys.exit(0)
    
    print("\n⏳ Предсказание начато...\n")
    
    # Логирование предсказания
    predict_log = f'logs/predict_{timestamp}.log'
    print(f"📝 Логи предсказания сохраняются в: {predict_log}\n")
    
    exit_code = os.system(f'python predict.py 2>&1 | tee {predict_log}')
    
    if exit_code != 0:
        print(f"\n❌ ОШИБКА при предсказании!")
        print(f"📋 Проверьте лог: {predict_log}")
        sys.exit(1)
    
    if not os.path.exists('predicted.las'):
        print(f"\n❌ Файл predicted.las не был создан! Проверьте ошибки.")
        print(f"📋 Проверьте лог: {predict_log}")
        sys.exit(1)
    
    print("\n✅ Предсказание завершено!")
    
    # Финал
    print("\n" + "="*60)
    print("  ✨ ВСЕ ГОТОВО!")
    print("="*60)
    print("\n📁 Созданные файлы:")
    print("  • checkpoints/best_model.pth - обученная модель")
    print("  • predicted.las - размеченное облако точек")
    print(f"  • {train_log} - лог обучения")
    print(f"  • {predict_log} - лог предсказания")
    
    print("\n💡 Что дальше?")
    print("  • Визуализировать результат: python visualize.py predicted.las")
    print("  • Сравнить с оригиналом: python main.py (пункт 5)")

if __name__ == '__main__':
    main()