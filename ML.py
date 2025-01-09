import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# Função de ativação radial (Gaussiana)
def radial_basis_function(x, c, sigma):
    return np.exp(-cdist(x, c)**2 / (2 * sigma**2))

# Classe RBF
class RBF:
    def __init__(self, num_centers=4, sigma=1.0):
        self.num_centers = num_centers
        self.sigma = sigma
        self.centers = None
        self.weights = None

    def fit(self, X, y):
        # Escolher centros aleatórios do conjunto de dados
        self.centers = X[np.random.choice(X.shape[0], self.num_centers, replace=False)]
        
        # Calcular matriz Phi (ativações das funções radiais)
        Phi = radial_basis_function(X, self.centers, self.sigma)
        
        # Ajustar os pesos (solução linear)
        self.weights = np.linalg.pinv(Phi) @ y

    def predict(self, X):
        # Calcular a matriz Phi para os dados de entrada
        Phi = radial_basis_function(X, self.centers, self.sigma)
        # Retornar predições
        return Phi @ self.weights

# Carregar os dados
try:
    data_train = np.loadtxt("dados_01.dat")
    data_test = np.loadtxt("dados_02.dat")
except Exception as e:
    print(f"Erro ao carregar os arquivos: {e}")
    exit()

# Extração dos sinais
t_train = data_train[:, 0]  # Tempo (não usado)
u_train = data_train[:, 1]  # Entrada
y_train = data_train[:, 2]  # Saída

t_test = data_test[:, 0]  # Tempo (não usado)
u_test = data_test[:, 1]  # Entrada
y_test = data_test[:, 2]  # Saída

# Preparar os dados com base nos atrasos
def prepare_data_with_delays(u, y):
    Y = np.array([np.concatenate((np.zeros(2), y[:-2])),
                  np.concatenate((np.zeros(1), y[:-1])),
                  u,
                  np.concatenate((np.zeros(1), u[:-1]))]).T
    return Y

# Criar as matrizes de entrada
Y_train = prepare_data_with_delays(u_train, y_train)
Y_test = prepare_data_with_delays(u_test, y_test)

# Treinar a Rede RBF
rbf = RBF(num_centers=4, sigma=1.0)
rbf.fit(Y_train, y_train)

# Prever os dados de teste
y_test_pred = rbf.predict(Y_test)

# Plotar os resultados
plt.figure(figsize=(14, 4))
plt.plot(t_test, y_test, "b.", label="Dados 02 (Reais)")  # Bolinhas para os dados reais
plt.plot(t_test, y_test_pred, "--r", label="Ajuste (teste)")  # Linha tracejada para predição
plt.xlabel("Tempo")
plt.ylabel("Saída")
plt.title("Validação da Rede RBF com Dados de Teste")
plt.legend()
plt.show()
