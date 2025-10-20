import laspy
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_prediction(ground_truth_las, predicted_las):
    """Оценка качества предсказания"""
    print("="*60)
    print("  📊 ОЦЕНКА КАЧЕСТВА ПРЕДСКАЗАНИЯ")
    print("="*60)
    
    # Загрузка файлов
    print("\n📂 Загрузка файлов...")
    gt = laspy.read(ground_truth_las)
    pred = laspy.read(predicted_las)
    
    # Извлечение меток
    y_true = gt.classification
    y_pred = pred.classification
    
    # Проверка размеров
    if len(y_true) != len(y_pred):
        print(f"❌ Ошибка: Разное количество точек!")
        print(f"   Ground truth: {len(y_true)}")
        print(f"   Predicted: {len(y_pred)}")
        return
    
    print(f"✅ Загружено {len(y_true)} точек")
    
    # Общая точность
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\n🎯 Общая точность: {accuracy*100:.2f}%")
    
    # Per-class метрики
    print("\n📈 Метрики по классам:")
    print(classification_report(y_true, y_pred, 
                               labels=list(range(1, 9)),
                               target_names=[f'Class {i}' for i in range(1, 9)],
                               digits=4))
    
    # Confusion Matrix
    print("\n📊 Матрица ошибок:")
    cm = confusion_matrix(y_true, y_pred, labels=list(range(1, 9)))
    
    # Нормализованная матрица
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Визуализация
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Абсолютные значения
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=range(1, 9), yticklabels=range(1, 9))
    ax1.set_title('Confusion Matrix (Counts)')
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')
    
    # Нормализованные значения
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', ax=ax2,
                xticklabels=range(1, 9), yticklabels=range(1, 9))
    ax2.set_title('Confusion Matrix (Normalized)')
    ax2.set_ylabel('True Label')
    ax2.set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("\n💾 Матрица ошибок сохранена: confusion_matrix.png")
    plt.show()
    
    # Статистика по классам
    print("\n📊 Детальная статистика:")
    for i in range(1, 9):
        mask_true = y_true == i
        mask_pred = y_pred == i
        
        n_true = mask_true.sum()
        n_pred = mask_pred.sum()
        n_correct = (y_true[mask_true] == y_pred[mask_true]).sum()
        
        if n_true > 0:
            recall = n_correct / n_true
            precision = n_correct / n_pred if n_pred > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"\n  Класс {i}:")
            print(f"    Ground truth: {n_true} точек")
            print(f"    Predicted: {n_pred} точек")
            print(f"    Correct: {n_correct} точек")
            print(f"    Precision: {precision*100:.2f}%")
            print(f"    Recall: {recall*100:.2f}%")
            print(f"    F1-Score: {f1*100:.2f}%")

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) == 3:
        evaluate_prediction(sys.argv[1], sys.argv[2])
    else:
        print("Использование: python evaluate.py <ground_truth.las> <predicted.las>")
        print("\nПример:")
        print("  python evaluate.py Univer2019.las predicted.las")