import numpy as np
from PIL import Image

# Параметры модели
C_max = 255  # Максимальная интенсивность пикселя
r, m = 8, 8  # Размер прямоугольников (8x8 пикселей)
Q = r*m*3  # Общее количество прямоугольников для 256x256 изображения
learning_rate = 0.01  # Начальный коэффициент обучения
epochs = 1000  # Количество эпох обучения
max_loss_threshold = 1000  # Установите максимальное значение ошибки

# Функция для загрузки и предобработки изображения
def load_and_preprocess_image(file_path):
    # Загрузка изображения и конвертация в grayscale
    image = Image.open(file_path).convert("L")  # Конвертация в черно-белый режим
    image = image.resize((256, 256))  # Изменение размера на 256x256
    
    # Преобразование изображения в массив numpy
    pixel_data = np.array(image, dtype=np.float32)
    
    # Нормализация пикселей в диапазон [-1, 1]
    pixel_data = (2 * pixel_data / C_max) - 1
    return pixel_data

# Функция для разделения изображения на прямоугольники r x m
def split_into_blocks(pixel_data, r, m):
    blocks = []
    for i in range(0, pixel_data.shape[0], r):
        for j in range(0, pixel_data.shape[1], m):
            block = pixel_data[i:i+r, j:j+m]
            blocks.append(block.flatten())  # Преобразование вектора
    return np.array(blocks)

# Функция для создания линейной рециркуляционной сети
class RecirculationNetwork:
    def __init__(self, input_dim, hidden_dim):
        limit = np.sqrt(6. / (input_dim + hidden_dim))
        # self.Wf = np.random.uniform(-limit, limit, (input_dim, hidden_dim))
        # self.Wb = np.random.uniform(-limit, limit, (hidden_dim, input_dim))
        self.Wf = np.random.randn(input_dim, hidden_dim)   # Веса прямого распространения
        self.Wb = np.random.randn(input_dim, hidden_dim)  # Веса обратного распространения
        
        print(self.Wf)
    def forward(self, X):
        return np.dot(self.Wf, X)  # Прямое распространение

    def backward(self, Y):
        return np.dot(self.Wb, Y)  # Обратное распространение

    def train(self, X, epochs, learning_rate):
        
        for epoch in range(epochs):
            total_loss = 0
            for x in X:
                # Прямое распространение
                y = self.forward(x)
                
                # Обратное распространение
                x_reconstructed = self.backward(y)
                
                # Вычисление ошибки
                loss = np.mean((x - x_reconstructed) ** 2)
                total_loss += loss
                
                # Обновление весов с адаптивным коэффициентом обучения
                self.Wf += learning_rate * np.outer(y, x)
                self.Wb += learning_rate * np.outer(x, y)
            
            # Вывод ошибки каждые 100 эпох
            if epoch % 100 == 0:
                average_loss = total_loss / len(X)
                print(f"Эпоха {epoch}, ошибка: {average_loss}, коэффициент обучения: {learning_rate}")
                print("Input DIM raven:",input_dim)
                if average_loss > max_loss_threshold:
                    print("Ошибка превышает максимальное значение!")
            
            # Адаптивное уменьшение коэффициента обучения
            learning_rate *= 0.99  # Например, на 1% на каждую эпоху

# Функция для восстановления изображения после обучения
def reconstruct_image(network, blocks, r, m):
    reconstructed_blocks = []
    for block in blocks:
        y = network.forward(block)
        reconstructed_block = network.backward(y)
        reconstructed_blocks.append(reconstructed_block.reshape(r, m))
    
    # Собираем прямоугольники обратно в изображение
    reconstructed_image = np.zeros((256, 256))
    idx = 0
    for i in range(0, 256, r):
        for j in range(0, 256, m):
            reconstructed_image[i:i+r, j:j+m] = reconstructed_blocks[idx]
            idx += 1
            
    # Обратное преобразование к диапазону [0, 255]
    reconstructed_image = ((reconstructed_image + 1) / 2 * C_max).clip(0, 255)
    return Image.fromarray(reconstructed_image.astype(np.uint8))

# Основной код
if __name__ == "__main__":
    # Загрузка и предобработка изображения
    file_path = "input_image.bmp"
    pixel_data = load_and_preprocess_image(file_path)
    
    # Разделение изображения на блоки
    blocks = split_into_blocks(pixel_data, r, m)
    
    # Инициализация и обучение сети
    input_dim = r * m * 3 
    hidden_dim = 64  # Размер скрытого слоя можно настроить
    network = RecirculationNetwork(input_dim, hidden_dim)
    network.train(blocks, epochs, learning_rate)
    
    # Восстановление изображения
    reconstructed_image = reconstruct_image(network, blocks, r, m)
    reconstructed_image.show()
    reconstructed_image.save("obrabotanniy.bmp")
