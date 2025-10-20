"""
Просмотр логов обучения
"""
import os
from datetime import datetime

def view_logs():
    """Показать последние логи"""
    print("="*60)
    print("  📋 ПРОСМОТР ЛОГОВ")
    print("="*60)
    
    logs_dir = 'logs'
    
    if not os.path.exists(logs_dir):
        print("\n❌ Папка logs не найдена!")
        return
    
    log_files = [f for f in os.listdir(logs_dir) if f.endswith('.log')]
    
    if not log_files:
        print("\n❌ Лог-файлы не найдены!")
        return
    
    # Сортировка по времени изменения
    log_files.sort(key=lambda x: os.path.getmtime(os.path.join(logs_dir, x)), reverse=True)
    
    print(f"\n📁 Найдено логов: {len(log_files)}\n")
    
    for i, log_file in enumerate(log_files[:10], 1):
        path = os.path.join(logs_dir, log_file)
        size = os.path.getsize(path) / 1024  # KB
        mtime = datetime.fromtimestamp(os.path.getmtime(path))
        print(f"{i}. {log_file}")
        print(f"   Размер: {size:.2f} KB")
        print(f"   Изменен: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
    
    choice = input("Введите номер лога для просмотра (или Enter для последнего): ").strip()
    
    if choice == '':
        selected = log_files[0]
    else:
        try:
            idx = int(choice) - 1
            selected = log_files[idx]
        except:
            print("❌ Неверный выбор!")
            return
    
    print(f"\n{'='*60}")
    print(f"  📄 {selected}")
    print(f"{'='*60}\n")
    
    with open(os.path.join(logs_dir, selected), 'r', encoding='utf-8') as f:
        content = f.read()
        
        # Показываем последние 100 строк
        lines = content.split('\n')
        if len(lines) > 100:
            print(f"... ({len(lines) - 100} строк пропущено) ...\n")
            print('\n'.join(lines[-100:]))
        else:
            print(content)
    
    # Поиск ошибок
    if 'ERROR' in content or 'Traceback' in content or '❌' in content:
        print(f"\n{'='*60}")
        print("  ⚠️  НАЙДЕНЫ ОШИБКИ В ЛОГЕ")
        print(f"{'='*60}\n")
        
        errors = []
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'ERROR' in line or 'Traceback' in line or '❌' in line:
                # Показываем контекст ошибки
                start = max(0, i - 2)
                end = min(len(lines), i + 5)
                errors.append('\n'.join(lines[start:end]))
        
        for i, error in enumerate(errors[:5], 1):  # Показываем первые 5 ошибок
            print(f"\n--- Ошибка {i} ---")
            print(error)
            print()

if __name__ == '__main__':
    view_logs()