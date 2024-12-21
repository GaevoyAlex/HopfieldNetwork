import numpy as np

class HopfieldNetwork:
    def __init__(self, input_size: int):

        self.input_size = input_size
        self.weights = np.zeros((input_size, input_size))  
        self.thresholds = np.zeros(input_size)  

    def activation(self, x: np.ndarray) -> np.ndarray:
        """гиперболический тангенс"""
        return np.tanh(x)

    def calculate_energy(self, state: np.ndarray) -> float:
        return -0.5 * np.dot(np.dot(state, self.weights), state) + np.dot(state, self.thresholds)

    def update_state_sync(self, state: np.ndarray) -> np.ndarray:
        """
        yi(t+1) = F(∑ωijyj(t) - Ti) 
        """
        net_input = np.dot(self.weights, state) - self.thresholds
        return self.activation(net_input)

    def train(self, patterns: list[np.ndarray]) -> None:
        """
        Обучение по правилу Хебба
        ωij = ∑(xi^k * xj^k), где i≠j
        ωii = 0
        """
        
        if len(patterns) > 0.15 * self.input_size:
            print("Количество образов превышает теоретическую емкость")
            print(len(patterns) ,'Количество паттернов')
            print(self.input_size*0.15, 'Максимально допустимое число обрвзов')
        
        else:
            X = np.array(patterns)
            n_patterns = len(patterns)

            for i in range(self.input_size):
                for j in range(self.input_size):
                    if i != j: 
                        self.weights[i, j] = sum(X[k, i] * X[k, j] for k in range(n_patterns))

            # self.weights /= n_patterns
            
            self.thresholds = np.mean(X, axis=0)
            

    def recognize(self, pattern: np.ndarray, max_iterations) -> np.ndarray:
        current_state = pattern.copy()
        previous_state = None
        prev_prev_state = None

        for i in range(max_iterations):
            new_state = self.update_state_sync(current_state)

            energy = self.calculate_energy(new_state)
            print(f"Iteration {i}, Energy: {energy}")

            if prev_prev_state is not None and np.allclose(new_state, prev_prev_state):
                print(f"Найден устойчивый цикл на итерации {i}")
                return new_state

            prev_prev_state = previous_state
            previous_state = current_state
            current_state = new_state

        return current_state