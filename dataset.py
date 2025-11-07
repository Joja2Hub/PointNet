import torch
from torch.utils.data import Dataset
import laspy
import numpy as np

class LASDataset(Dataset):
    """Датасет для LAS файлов"""
    def __init__(self, las_file, num_points=4096, block_size=50.0, stride=25.0, train=True):
        self.num_points = num_points
        self.block_size = block_size
        self.stride = stride
        self.train = train
        
        # Загрузка LAS файла
        print(f"📂 Загрузка {las_file}...")
        las = laspy.read(las_file)
        
        # Извлечение координат
        self.points = np.vstack([las.x, las.y, las.z]).T.astype(np.float32)
        
        # Извлечение меток (если есть)
        try:
            self.labels = np.array(las.classification).astype(np.int64) - 1  # Классы 1-8 -> 0-7
            self.has_labels = True
            
            # Проверяем какие классы действительно есть в данных
            unique_labels = np.unique(self.labels)
            print(f"🏷️  Классы в LAS файле: {unique_labels + 1}")
            print(f"📊 Количество точек по классам:")
            for class_id in unique_labels:
                count = np.sum(self.labels == class_id)
                percentage = 100.0 * count / len(self.labels)
                print(f"   Класс {class_id + 1}: {count} точек ({percentage:.2f}%)")
                
        except Exception as e:
            print(f"⚠️  Ошибка при чтении меток: {e}")
            self.labels = np.zeros(len(self.points), dtype=np.int64)
            self.has_labels = False
        
        # Нормализация координат
        self.coord_min = np.min(self.points, axis=0)
        self.coord_max = np.max(self.points, axis=0)
        
        # Разбиение на блоки
        self._create_blocks()
        
        print(f"✅ Загружено {len(self.points)} точек")
        print(f"📦 Создано {len(self.blocks)} блоков")
        
        # Анализ распределения классов в блоках
        self._analyze_block_classes()
    
    def _analyze_block_classes(self):
        """Анализ каких классов в блоках"""
        if not self.has_labels:
            return
            
        print(f"🔍 Анализ классов в первых 10 блоках:")
        for i in range(min(10, len(self.blocks))):
            indices = self.blocks[i]
            block_labels = self.labels[indices]
            unique_classes = np.unique(block_labels)
            print(f"   Блок {i}: классы {[c + 1 for c in unique_classes.tolist()]}")
    
    def _create_blocks(self):
        """Разбиение облака точек на блоки"""
        self.blocks = []
        
        x_min, y_min, _ = self.coord_min
        x_max, y_max, _ = self.coord_max
        
        # Сетка блоков
        x_blocks = int(np.ceil((x_max - x_min - self.block_size) / self.stride)) + 1
        y_blocks = int(np.ceil((y_max - y_min - self.block_size) / self.stride)) + 1
        
        for i in range(x_blocks):
            for j in range(y_blocks):
                x_start = x_min + i * self.stride
                y_start = y_min + j * self.stride
                x_end = x_start + self.block_size
                y_end = y_start + self.block_size
                
                # Поиск точек в блоке
                mask = (self.points[:, 0] >= x_start) & (self.points[:, 0] <= x_end) & \
                       (self.points[:, 1] >= y_start) & (self.points[:, 1] <= y_end)
                
                indices = np.where(mask)[0]
                
                if len(indices) > 100:  # Минимум точек в блоке
                    self.blocks.append(indices)
    
    def __len__(self):
        return len(self.blocks)
    
    def __getitem__(self, idx):
        """Получение блока точек"""
        indices = self.blocks[idx]
        
        # Сэмплирование или дополнение до num_points
        if len(indices) >= self.num_points:
            choice = np.random.choice(len(indices), self.num_points, replace=False)
        else:
            choice = np.random.choice(len(indices), self.num_points, replace=True)
        
        selected_indices = indices[choice]
        
        # Получение точек и меток
        points = self.points[selected_indices].copy()
        labels = self.labels[selected_indices].copy()
        
        # Нормализация блока (центрирование)
        centroid = np.mean(points, axis=0)
        points = points - centroid
        
        # Дополнительная нормализация по максимальному расстоянию
        max_dist = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
        if max_dist > 0:
            points = points / max_dist
        
        # Аугментация данных (только для тренировки)
        if self.train:
            points = self._augment(points)
        
        # Преобразование в тензоры
        points = torch.from_numpy(points).float()
        labels = torch.from_numpy(labels).long()
        
        # Транспонирование для модели (N, 3) -> (3, N)
        points = points.transpose(0, 1)
        
        return points, labels
    
    def _augment(self, points):
        """Аугментация данных"""
        # Случайное вращение вокруг оси Z
        theta = np.random.uniform(0, 2 * np.pi)
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
        points = points @ rotation_matrix.T
        
        # Случайное масштабирование
        scale = np.random.uniform(0.8, 1.2)
        points = points * scale
        
        # Случайный джиттер
        jitter = np.random.normal(0, 0.01, points.shape)
        points = points + jitter
        
        return points