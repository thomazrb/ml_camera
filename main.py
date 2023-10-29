import cv2
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Capturar vídeo da webcam
print('Inicializando Camera...')
cap = cv2.VideoCapture(0)

# Carregar o conjunto de dados MNIST. MINIST ( http://yann.lecun.com/exdb/mnist/ ) 
# é um conjunto de dados contendo imagens de dígitos escritos a mão, que é
# possível baixar diretamente nos datasets do scikit learn através do fetch_openml
print('Carregando base de dados...')
mnist = fetch_openml('mnist_784', parser='auto')
X, y = mnist.data, mnist.target.astype(int)

# Função para redimensionar as imagens da webcam para 28x28 pixels para condizer com as imagens da base de dados
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (28, 28))
    gray = np.array(gray).flatten()
    return gray

X = X.astype(float) / 255.0  # Normalizar os valores de pixel

# Dividir os dados em conjuntos de treinamento e teste
print('Separando os dados em treino e teste...')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar um modelo simples (Regressão Logística)
print('Treinando o modelo...')
model = LogisticRegression(solver='lbfgs', max_iter=100)
model.fit(X_train, y_train)

print('Executando...')
while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Pré-processamento da imagem
    gray = preprocess_image(frame)

    # Fazer uma previsão com o modelo treinado
    prediction = model.predict([gray])

    cv2.putText(frame, str(prediction[0]), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 2)
    cv2.imshow('Reconhecimento de Digitos', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): # Tecla q para fechar a janela.
        break

cap.release()
cv2.destroyAllWindows()
