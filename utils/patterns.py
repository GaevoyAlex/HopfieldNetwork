import numpy as np

def create_letter_patterns(size: int = 8) -> dict:
    """Создание паттернов букв и цифр"""
    patterns = {}
    
    # A
    def create_A(size):
        pattern = np.ones((size, size)) * -1
        for i in range(size):
            pattern[i, 0] = 1  
            pattern[i, size-1] = 1  
        pattern[0, :] = 1  
        pattern[size//2, :] = 1  
        return pattern.flatten()
    
    # X
    def create_X(size):
        pattern = np.ones((size, size)) * -1
        for i in range(size):
            pattern[i, i] = 1  
            pattern[i, size-1-i] = 1
        return pattern.flatten()
    
    # H
    def create_H(size):
        pattern = np.ones((size, size)) * -1
        pattern[:, 0] = 1  
        pattern[:, size-1] = 1  
        pattern[size//2, :] = 1  
        return pattern.flatten()

    # O
    def create_letter_O(size):
        pattern = np.ones((size, size)) * -1
        pattern[0, :] = 1  
        pattern[size-1, :] = 1  
        pattern[:, 0] = 1  
        pattern[:, size-1] = 1  
        return pattern.flatten()

    # T
    def create_letter_T(size):
        pattern = np.ones((size, size)) * -1
        pattern[0, :] = 1  
        pattern[:, size//2] = 1  
        return pattern.flatten()
    
    # 0
    def create_0(size):
        pattern = np.ones((size, size)) * -1
        pattern[0, :] = 1
        pattern[size-1, :] = 1
        pattern[:, 0] = 1
        pattern[:, size-1] = 1
        return pattern.flatten()
    
    # 1
    def create_1(size):
        pattern = np.ones((size, size)) * -1
        pattern[:, size//2] = 1
        pattern[size-1, size//2-1:size//2+2] = 1  # основание
        return pattern.flatten()
    
    # 2
    def create_2(size):
        pattern = np.ones((size, size)) * -1
        pattern[0, :] = 1  # верх
        pattern[size-1, :] = 1  # низ
        pattern[size//2, :] = 1  # середина
        pattern[0:size//2, size-1] = 1  # правая верхняя часть
        pattern[size//2:, 0] = 1  # левая нижняя часть
        return pattern.flatten()
    
    # 3
    def create_3(size):
        pattern = np.ones((size, size)) * -1
        pattern[0, :] = 1  # верх
        pattern[size-1, :] = 1  # низ
        pattern[size//2, :] = 1  # середина
        pattern[:, size-1] = 1  # правая сторона
        return pattern.flatten()
    
    # 4
    def create_4(size):
        pattern = np.ones((size, size)) * -1
        pattern[:size//2+1, 0] = 1  # левая верхняя часть
        pattern[size//2, :] = 1  # горизонтальная линия
        pattern[:, size-1] = 1  # правая вертикальная линия
        return pattern.flatten()
    
    # 5
    def create_5(size):
        pattern = np.ones((size, size)) * -1
        pattern[0, :] = 1  # верх
        pattern[size-1, :] = 1  # низ
        pattern[size//2, :] = 1  # середина
        pattern[0:size//2, 0] = 1  # левая верхняя часть
        pattern[size//2:, size-1] = 1  # правая нижняя часть
        return pattern.flatten()
    
    # 6
    def create_6(size):
        pattern = np.ones((size, size)) * -1
        pattern[0, :] = 1  # верх
        pattern[size-1, :] = 1  # низ
        pattern[size//2, :] = 1  # середина
        pattern[:, 0] = 1  # левая сторона
        pattern[size//2:, size-1] = 1  # правая нижняя часть
        return pattern.flatten()
    
    # 7
    def create_7(size):
        pattern = np.ones((size, size)) * -1
        pattern[0, :] = 1  # верх
        pattern[:, size-1] = 1  # правая сторона
        return pattern.flatten()
    
    # 8
    def create_8(size):
        pattern = np.ones((size, size)) * -1
        pattern[0, :] = 1  # верх
        pattern[size-1, :] = 1  # низ
        pattern[size//2, :] = 1  # середина
        pattern[:, 0] = 1  # левая сторона
        pattern[:, size-1] = 1  # правая сторона
        return pattern.flatten()
    
    # 9
    def create_9(size):
        pattern = np.ones((size, size)) * -1
        pattern[0, :] = 1  # верх
        pattern[size-1, :] = 1  # низ
        pattern[size//2, :] = 1  # середина
        pattern[:size//2, 0] = 1  # левая верхняя часть
        pattern[:, size-1] = 1  # правая сторона
        return pattern.flatten()

    # Добавляем буквы в словарь
    patterns['A'] = create_A(size)
    patterns['X'] = create_X(size)
    patterns['H'] = create_H(size)
    patterns['O'] = create_letter_O(size)
    patterns['T'] = create_letter_T(size)
    
    # Добавляем цифры в словарь
    for i in range(10):
        patterns[str(i)] = eval(f'create_{i}(size)')
    
    return patterns