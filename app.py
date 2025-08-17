
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from model.network import HopfieldNetwork
from utils.patterns import create_letter_patterns


def add_noise(pattern: np.ndarray, noise_level: float = 0.2) -> np.ndarray:
    """Добавление шума к образу"""
    noisy = pattern.copy()
    mask = np.random.random(len(pattern)) < noise_level
    noisy[mask] = 0
    return noisy

def plot_pattern(pattern: np.ndarray, size: int, title: str = '') -> plt.Figure:
    """Отображение паттерна"""
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(pattern.reshape((size, size)), cmap='binary')
    ax.set_title(title)
    ax.grid(True)
    ax.axis('off')
    return fig


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

def initialize_session_state():
    """Инициализация состояния сессии"""
    if 'network' not in st.session_state:
        st.session_state.network = None
    if 'trained' not in st.session_state:
        st.session_state.trained = False
    if 'selected_patterns' not in st.session_state:
        st.session_state.selected_patterns = []
    if 'weights' not in st.session_state:
        st.session_state.weights = None
    if 'thresholds' not in st.session_state:
        st.session_state.thresholds = None

def train_network():
    """Функция для обучения сети"""
    network = HopfieldNetwork(st.session_state.pattern_size * st.session_state.pattern_size)
    network.train(st.session_state.selected_patterns)
    
    # Сохраняем веса и пороги
    st.session_state.weights = network.weights.copy()
    st.session_state.thresholds = network.thresholds.copy()
    st.session_state.network = network
    st.session_state.trained = True
    
def main():
    st.title('Сеть Хопфилда: Распознавание букв')
    
    initialize_session_state()

    pattern_size = st.sidebar.slider('Размер паттерна', 3, 12, 8)
    st.session_state.pattern_size = pattern_size
    noise_level = st.sidebar.slider('Уровень шума', 0.0, 0.5, 0.2)
    num_iterations = st.sidebar.slider("Колличество итераций",0,200,50)
    patterns = create_letter_patterns(pattern_size)
    
    st.header('Выбор образов для обучения')
    selected_letters = st.multiselect(
        'Выберите буквы для обучения',
        list(patterns.keys()),
        default=['A', 'X', 'H']
    )
    
    if selected_letters:
        
        st.subheader('Выбранные образы:')
        cols = st.columns(len(selected_letters))
        
        selected_patterns = []
        for i, letter in enumerate(selected_letters):
            pattern = patterns[letter]
            selected_patterns.append(pattern)
            with cols[i]:
                fig = plot_pattern(pattern, pattern_size, f'Буква {letter}')
                st.pyplot(fig)
                plt.close()
        
        st.session_state.selected_patterns = selected_patterns
        
        
        train_button = st.button('Обучить сеть')
        if train_button:
            train_network()
            
            st.success('Сеть обучена!')
            
            
            st.subheader('Матрица весов')
            fig, ax = plt.subplots(figsize=(6, 6))
            im = ax.imshow(st.session_state.weights, cmap='RdBu')
            plt.colorbar(im)
            st.pyplot(fig)
            plt.close()
        
        if st.session_state.trained:
            st.header('Тестирование сети')
            test_col1, test_col2 = st.columns(2)
            
            with test_col1:
                test_letter = st.selectbox(
                    'Выберите букву для тестирования',
                    selected_letters
                )
            
            with test_col2:
                test_button = st.button('Тестировать')
            
            if test_letter and test_button:
                # Восстанавливаем сеть из сохраненных весов
                network = HopfieldNetwork(pattern_size * pattern_size)
                network.weights = st.session_state.weights.copy()
                network.thresholds = st.session_state.thresholds.copy()
                
                # Тестов образ ()
                test_pattern = patterns[test_letter]
                noisy_pattern = add_noise(test_pattern, noise_level)
                
                # Распознавание
                recognized_pattern = network.recognize(noisy_pattern,max_iterations=num_iterations)
                
                # Визуализация
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**Исходный образ**")
                    fig = plot_pattern(test_pattern, pattern_size, f'Буква {test_letter}')
                    st.pyplot(fig)
                    plt.close()
                
                with col2:
                    st.markdown("**Зашумленный образ**")
                    fig = plot_pattern(noisy_pattern, pattern_size, f'Шум: {noise_level:.1%}')
                    st.pyplot(fig)
                    plt.close()
                
                with col3:
                    st.markdown("**Распознанный образ**")
                    fig = plot_pattern(recognized_pattern, pattern_size, 'Результат')
                    st.pyplot(fig)
                    plt.close()
                
                # Определение наиболее похожего образа
                similarities = [np.corrcoef(recognized_pattern, p)[0, 1] 
                              for p in selected_patterns]
                most_similar_idx = np.argmax(similarities)
                similarity = similarities[most_similar_idx]
                
                # Вывод результатов
                st.info(f"""
                    Результаты распознавания:
                    - Распознан как буква: {selected_letters[most_similar_idx]}
                    - Схожесть с оригиналом: {similarity:.2%}
                    - Энергия конечного состояния: {network.calculate_energy(recognized_pattern):.2f}
                """)

if __name__ == '__main__':
    main()