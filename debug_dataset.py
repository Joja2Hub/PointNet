"""
Отладочный скрипт для проверки датасета
"""
import traceback
from dataset import LASDataset
import torch

def test_dataset():
    """Тестирование загрузки датасета"""
    print("="*60)
    print("  🔍 ОТЛАДКА ДАТАСЕТА")
    print("="*60)
    
    try:
        print("\n1️⃣ Создание датасета...")
        dataset = LASDataset(
            'Univer2019.las',
            num_points=4096,
            block_size=50.0,
            stride=25.0,
            train=True
        )
        print(f"✅ Датасет создан: {len(dataset)} блоков")
        
        if len(dataset) == 0:
            print("❌ Датасет пустой!")
            return
        
        print("\n2️⃣ Тест загрузки первого блока...")
        try:
            points, labels = dataset[0]
            print(f"✅ Блок загружен:")
            print(f"   Points shape: {points.shape}")
            print(f"   Labels shape: {labels.shape}")
            print(f"   Points dtype: {points.dtype}")
            print(f"   Labels dtype: {labels.dtype}")
            print(f"   Unique labels: {torch.unique(labels).tolist()}")
        except Exception as e:
            print(f"❌ Ошибка загрузки блока:")
            print(traceback.format_exc())
            return
        
        print("\n3️⃣ Тест загрузки нескольких блоков...")
        num_test = min(10, len(dataset))
        errors = 0
        for i in range(num_test):
            try:
                points, labels = dataset[i]
                print(f"   Блок {i}: ✅ ({points.shape})")
            except Exception as e:
                print(f"   Блок {i}: ❌ {str(e)}")
                errors += 1
        
        if errors > 0:
            print(f"\n⚠️  Ошибок: {errors}/{num_test}")
        else:
            print(f"\n✅ Все блоки загружены успешно!")
        
        print("\n4️⃣ Тест DataLoader...")
        from torch.utils.data import DataLoader
        
        loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0
        )
        
        try:
            points_batch, labels_batch = next(iter(loader))
            print(f"✅ Батч загружен:")
            print(f"   Points batch shape: {points_batch.shape}")
            print(f"   Labels batch shape: {labels_batch.shape}")
        except Exception as e:
            print(f"❌ Ошибка загрузки батча:")
            print(traceback.format_exc())
            return
        
        print("\n✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ!")
        
    except Exception as e:
        print(f"\n❌ КРИТИЧЕСКАЯ ОШИБКА:")
        print(traceback.format_exc())

if __name__ == '__main__':
    test_dataset()