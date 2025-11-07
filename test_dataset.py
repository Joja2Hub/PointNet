# test_dataset.py
import torch
from dataset import LASDataset

def test_dataset():
    dataset = LASDataset(
        "datasets/raw/Univer2019.las",
        num_points=4096,
        block_size=50.0,
        stride=25.0,
        train=True
    )
    
    print(f"Всего блоков: {len(dataset)}")
    
    # Проверим первые 20 блоков
    for i in range(min(20, len(dataset))):
        points, labels = dataset[i]
        unique = torch.unique(labels)
        print(f"Блок {i}: классы {[c + 1 for c in unique.tolist()]}, точек: {len(labels)}")

if __name__ == "__main__":
    test_dataset()