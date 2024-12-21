# test.py
import numpy as np
import matplotlib.pyplot as plt
from model.network import HopfieldNetwork

def create_test_patterns(size: int = 8) -> tuple[dict, list, list]:
    """
    Создание тестовых паттернов
    Returns:
        patterns: словарь всех паттернов
        train_patterns: список паттернов для обучения
        test_patterns: список паттернов для тестирования
    """
    patterns = {}
    
    # Буква A
    pattern_A = np.ones((size, size)) * -1
    for i in range(size):
        pattern_A[i, 0] = 1  # левая линия
        pattern_A[i, size-1] = 1  # правая линия
    pattern_A[0, :] = 1  # верхняя линия
    pattern_A[size//2, :] = 1  # средняя линия
    patterns['A'] = pattern_A.flatten()
    
    # Буква X
    pattern_X = np.ones((size, size)) * -1
    for i in range(size):
        pattern_X[i, i] = 1  # главная диагональ
        pattern_X[i, size-1-i] = 1  # побочная диагональ
    patterns['X'] = pattern_X.flatten()
    
    # Буква H
    pattern_H = np.ones((size, size)) * -1
    pattern_H[:, 0] = 1  # левая линия
    pattern_H[:, size-1] = 1  # правая линия
    pattern_H[size//2, :] = 1  # средняя линия
    patterns['H'] = pattern_H.flatten()
    
    # Выбираем паттерны для обучения и тестирования
    train_patterns = [patterns['A'], patterns['X']]  # Обучаем на A и X
    test_patterns = [patterns['A'], patterns['X'], patterns['H']]  # Тестируем все
    
    return patterns, train_patterns, test_patterns

def add_noise(pattern: np.ndarray, noise_level: float = 0.2) -> np.ndarray:
    noisy = pattern.copy()
    mask = np.random.random(len(pattern)) < noise_level
    noisy[mask] = 0
    return noisy

def plot_results(original, noisy, recognized, size, title=""):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
    
    ax1.imshow(original.reshape((size, size)), cmap='binary')
    ax1.set_title('Оригинал')
    ax1.axis('off')
    
    ax2.imshow(noisy.reshape((size, size)), cmap='binary')
    ax2.set_title('Зашумленный')
    ax2.axis('off')
    
    ax3.imshow(recognized.reshape((size, size)), cmap='binary')
    ax3.set_title('Распознанный')
    ax3.axis('off')
    
    plt.suptitle(title)
    plt.show()

def test_network():
    pattern_size = 8
    noise_levels = [0.1, 0.2, 0.3]

    patterns, train_patterns, test_patterns = create_test_patterns(pattern_size)
    letters = ['A', 'X', 'H']
    
    network = HopfieldNetwork(pattern_size * pattern_size)
    network.train(train_patterns)
    print(f"Сеть обучена на буквах: A, X")
    print(f"Размер входа: {pattern_size}x{pattern_size}")
    print(f"Теоретическая емкость памяти: {0.15 * pattern_size * pattern_size:.1f} образов")
    
    for idx, test_pattern in enumerate(test_patterns):
        letter = letters[idx]
        print(f"\nТестирование буквы {letter}:")
        
        for noise_level in noise_levels:
            print(f"\nУровень шума: {noise_level:.1%}")
            
            # Добавляем шум
            noisy_pattern = add_noise(test_pattern, noise_level)
            
            # Распознаем
            recognized_pattern = network.recognize(noisy_pattern)
            
            # Вычисляем схожесть с образцами
            similarities = [np.corrcoef(recognized_pattern, p)[0, 1] for p in train_patterns]
            most_similar_idx = np.argmax(similarities)
            similarity = similarities[most_similar_idx]
            recognized_as = ['A', 'X'][most_similar_idx]
            
            # Выводим результаты
            print(f"Распознано как: {recognized_as}")
            print(f"Схожесть: {similarity:.2%}")
            print(f"Энергия: {network.calculate_energy(recognized_pattern):.2f}")
            
            # Визуализируем результаты
            plot_results(
                test_pattern, 
                noisy_pattern, 
                recognized_pattern, 
                pattern_size,
                f"Буква {letter}, шум {noise_level:.1%}"
            )

if __name__ == "__main__":
    test_network()