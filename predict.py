import torch
import laspy
import numpy as np
from tqdm import tqdm
from model import PointNet2SemSeg

def predict_las_file(input_las, output_las, model_path, num_points=4096, batch_size=8):
    """
    Применение обученной модели к неразмеченному облаку точек
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Устройство: {device}")
    
    # Загрузка модели
    print(f"\n📥 Загрузка модели из {model_path}...")
    model = PointNet2SemSeg(num_classes=8).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("✅ Модель загружена")
    
    # Загрузка LAS файла
    print(f"\n📂 Загрузка {input_las}...")
    las = laspy.read(input_las)
    points = np.vstack([las.x, las.y, las.z]).T.astype(np.float32)
    num_total_points = len(points)
    print(f"✅ Загружено {num_total_points} точек")
    
    # Подготовка массива для предсказаний
    predictions = np.zeros(num_total_points, dtype=np.uint8)
    
    # Нормализация координат
    coord_min = np.min(points, axis=0)
    coord_max = np.max(points, axis=0)
    
    # Разбиение на блоки
    print("\n📦 Разбиение на блоки...")
    block_size = 50.0
    stride = 50.0  # Без перекрытия для предсказания
    
    blocks = []
    block_indices = []
    
    x_min, y_min, _ = coord_min
    x_max, y_max, _ = coord_max
    
    x_blocks = int(np.ceil((x_max - x_min) / stride))
    y_blocks = int(np.ceil((y_max - y_min) / stride))
    
    for i in range(x_blocks):
        for j in range(y_blocks):
            x_start = x_min + i * stride
            y_start = y_min + j * stride
            x_end = x_start + block_size
            y_end = y_start + block_size
            
            mask = (points[:, 0] >= x_start) & (points[:, 0] <= x_end) & \
                   (points[:, 1] >= y_start) & (points[:, 1] <= y_end)
            
            indices = np.where(mask)[0]
            
            if len(indices) > 0:
                blocks.append(points[indices])
                block_indices.append(indices)
    
    print(f"✅ Создано {len(blocks)} блоков")
    
     # Предсказание для каждого блока
    print("\n🔮 Предсказание классов...")
    
    with torch.no_grad():
        for block_pts, indices in tqdm(zip(blocks, block_indices), total=len(blocks)):
            # Обработка блока частями по num_points
            num_block_points = len(block_pts)
            
            if num_block_points == 0:
                continue
            
            # Центрирование блока
            centroid = np.mean(block_pts, axis=0)
            block_pts_centered = block_pts - centroid
            
            # Нормализация по максимальному расстоянию
            max_dist = np.max(np.sqrt(np.sum(block_pts_centered ** 2, axis=1)))
            if max_dist > 0:
                block_pts_centered = block_pts_centered / max_dist
            
            # Если точек больше num_points, обрабатываем частями
            if num_block_points > num_points:
                block_predictions = np.zeros(num_block_points, dtype=np.int64)
                
                # Разбиваем на части
                num_batches = int(np.ceil(num_block_points / num_points))
                
                for b in range(num_batches):
                    start_idx = b * num_points
                    end_idx = min((b + 1) * num_points, num_block_points)
                    
                    # Если последняя часть меньше num_points, дополняем
                    if end_idx - start_idx < num_points:
                        # Дополняем повторением случайных точек
                        current_pts = block_pts_centered[start_idx:end_idx]
                        num_missing = num_points - len(current_pts)
                        random_indices = np.random.choice(len(current_pts), num_missing, replace=True)
                        padded_pts = np.vstack([current_pts, current_pts[random_indices]])
                        
                        # Предсказание
                        pts_tensor = torch.from_numpy(padded_pts).float().unsqueeze(0).transpose(1, 2).to(device)
                        pred = model(pts_tensor)
                        pred_labels = pred.argmax(dim=2).cpu().numpy()[0]
                        
                        # Берем только реальные точки
                        block_predictions[start_idx:end_idx] = pred_labels[:len(current_pts)]
                    else:
                        pts_tensor = torch.from_numpy(block_pts_centered[start_idx:end_idx]).float().unsqueeze(0).transpose(1, 2).to(device)
                        pred = model(pts_tensor)
                        pred_labels = pred.argmax(dim=2).cpu().numpy()[0]
                        block_predictions[start_idx:end_idx] = pred_labels
            else:
                # Если точек меньше num_points, дополняем
                num_missing = num_points - num_block_points
                random_indices = np.random.choice(num_block_points, num_missing, replace=True)
                padded_pts = np.vstack([block_pts_centered, block_pts_centered[random_indices]])
                
                # Предсказание
                pts_tensor = torch.from_numpy(padded_pts).float().unsqueeze(0).transpose(1, 2).to(device)
                pred = model(pts_tensor)
                pred_labels = pred.argmax(dim=2).cpu().numpy()[0]
                
                # Берем только реальные точки
                block_predictions = pred_labels[:num_block_points]
            
            # Сохраняем предсказания (переводим 0-7 обратно в 1-8)
            predictions[indices] = block_predictions + 1
    
    # Сохранение результата
    print(f"\n💾 Сохранение результата в {output_las}...")
    
    # Создаем новый LAS файл с классификацией
    output_las_data = laspy.LasData(las.header)
    output_las_data.x = las.x
    output_las_data.y = las.y
    output_las_data.z = las.z
    
    # Копируем другие атрибуты если они есть
    if hasattr(las, 'intensity'):
        output_las_data.intensity = las.intensity
    if hasattr(las, 'return_number'):
        output_las_data.return_number = las.return_number
    if hasattr(las, 'number_of_returns'):
        output_las_data.number_of_returns = las.number_of_returns
    
    # Добавляем предсказанную классификацию
    output_las_data.classification = predictions
    
    # Сохраняем
    output_las_data.write(output_las)
    
    print("✅ Готово!")
    
    # Статистика предсказаний
    print("\n📊 Статистика предсказанных классов:")
    unique, counts = np.unique(predictions, return_counts=True)
    for class_id, count in zip(unique, counts):
        percentage = 100.0 * count / num_total_points
        print(f"   Класс {class_id}: {count} точек ({percentage:.2f}%)")

if __name__ == '__main__':
    # Параметры
    INPUT_LAS = 'unlabeled.las'
    OUTPUT_LAS = 'predicted.las'
    MODEL_PATH = 'checkpoints/best_model.pth'
    
    predict_las_file(INPUT_LAS, OUTPUT_LAS, MODEL_PATH)