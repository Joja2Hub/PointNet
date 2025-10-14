#!/usr/bin/env python3
"""
Универсальный анализ LAS-датасета.
Поддерживает любой LAS-файл, выводит статистику, проверяет наличие классификации.
"""

import os
import sys
import laspy
import numpy as np
from pathlib import Path

DATASETS_DIR = "datasets/row"

def list_las_files():
    """Возвращает список .las и .laz файлов в папке datasets/"""
    if not os.path.exists(DATASETS_DIR):
        print(f"❌ Папка '{DATASETS_DIR}' не найдена.")
        return []
    
    files = []
    for ext in ('*.las', '*.laz'):
        files.extend(Path(DATASETS_DIR).glob(ext))
    
    files = sorted([f.name for f in files])
    return files

def analyze_las_file(filepath):
    """Анализ одного LAS-файла"""
    print(f"\n{'='*70}")
    print(f"АНАЛИЗ ФАЙЛА: {filepath}")
    print("="*70)
    
    try:
        las = laspy.read(filepath)
        total_points = len(las.points)
        
        print(f"✅ Формат: LAS {las.header.version.major}.{las.header.version.minor}")
        print(f"📊 Всего точек: {total_points:,}")
        print(f"💾 Размер файла: {os.path.getsize(filepath) / (1024**2):.2f} MB")
        
        # Атрибуты
        dims = [dim.name for dim in las.point_format.dimensions]
        print(f"\n📋 Доступные атрибуты ({len(dims)}):")
        for dim in sorted(dims):
            print(f"  • {dim}")
        
        # Координаты
        print(f"\n🌍 Диапазоны координат:")
        print(f"  X: {las.x.min():.3f} → {las.x.max():.3f} (Δ={las.x.max() - las.x.min():.3f})")
        print(f"  Y: {las.y.min():.3f} → {las.y.max():.3f} (Δ={las.y.max() - las.y.min():.3f})")
        print(f"  Z: {las.z.min():.3f} → {las.z.max():.3f} (Δ={las.z.max() - las.z.min():.3f})")
        
        # Классификация
        has_classification = 'classification' in dims
        if has_classification:
            classes = las.classification
            unique, counts = np.unique(classes, return_counts=True)
            print(f"\n🏷️  Классификация обнаружена!")
            print(f"  Уникальных классов: {len(unique)}")
            print(f"  Диапазон: {unique.min()} → {unique.max()}")
            print(f"\n  Распределение по классам:")
            total = len(classes)
            for cls, cnt in zip(unique, counts):
                pct = cnt / total * 100
                print(f"    Класс {cls:2d}: {cnt:>8,} точек ({pct:>5.2f}%)")
        else:
            print(f"\n⚠️  Классификация отсутствует — файл подходит только для инференса или self-supervised обучения.")
        
        # Другие полезные поля
        optional_fields = ['intensity', 'return_number', 'number_of_returns', 'scan_angle_rank', 'gps_time']
        found_optional = [f for f in optional_fields if f in dims]
        if found_optional:
            print(f"\n🔍 Дополнительные поля: {', '.join(found_optional)}")
            for field in found_optional:
                arr = getattr(las, field)
                print(f"  • {field}: min={arr.min()}, max={arr.max()}, dtype={arr.dtype}")
        
        # Рекомендации
        print(f"\n💡 Рекомендации:")
        if has_classification:
            print(f"  → Подходит для семантической сегментации.")
            print(f"  → Количество классов для модели: {len(unique)} (но убедитесь, что классы корректны!)")
        else:
            print(f"  → Подходит только для инференса или генерации pseudo-labels.")
        
        if total_points > 10_000_000:
            print(f"  → Очень большой датасет! Рассмотрите уменьшение block_size или stride в dataset.py")
        elif total_points < 10_000:
            print(f"  → Очень маленький датасет! Возможно, недостаточно для обучения.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Ошибка при анализе файла '{filepath}': {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🔍 УНИВЕРСАЛЬНЫЙ АНАЛИЗ LAS-ДАТАСЕТОВ")
    print("="*70)
    
    # Создаём папку datasets, если её нет
    os.makedirs(DATASETS_DIR, exist_ok=True)
    
    files = list_las_files()
    
    if not files:
        print(f"\n📂 Папка '{DATASETS_DIR}' пуста или не содержит .las/.laz файлов.")
        print(f"💡 Поместите ваши LAS-файлы в папку '{DATASETS_DIR}' и повторите попытку.")
        sys.exit(1)
    
    print(f"\n📁 Найдено {len(files)} файлов в '{DATASETS_DIR}':")
    for i, f in enumerate(files, 1):
        size_mb = os.path.getsize(os.path.join(DATASETS_DIR, f)) / (1024**2)
        print(f"  {i}. {f} ({size_mb:.2f} MB)")
    
    print(f"\n❓ Выберите номер файла для анализа (1-{len(files)}), или 'all' для анализа всех:")
    choice = input("👉 Ваш выбор: ").strip()
    
    if choice.lower() == 'all':
        selected_files = files
    else:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(files):
                selected_files = [files[idx]]
            else:
                print("❌ Неверный номер.")
                sys.exit(1)
        except ValueError:
            print("❌ Неверный ввод.")
            sys.exit(1)
    
    success_count = 0
    for f in selected_files:
        filepath = os.path.join(DATASETS_DIR, f)
        if analyze_las_file(filepath):
            success_count += 1
    
    print(f"\n{'='*70}")
    print(f"✅ Успешно проанализировано: {success_count}/{len(selected_files)} файлов")
    if success_count == 0:
        print("❌ Все анализы завершились с ошибками.")
        sys.exit(1)

if __name__ == "__main__":
    main()