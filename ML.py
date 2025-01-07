import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

try:
    data = np.loadtxt("dados_01.dat")
    data2 = np.loadtxt("dados_02.dat")
except Exception as e:
    print(f"Erro ao carregar os arquivos: {e}")
    exit()

t = data[:, 0]  # Tempo
u = data[:, 1]  # Sinal de entrada
y = data[:, 2]  # Sinal de saída


plt.figure(figsize=(14, 4))
plt.plot(t, y, lw=2, alpha=0.5, label="Dados")
plt.xlabel("Tempo (s)")
plt.ylabel("Sinal de Saída")
plt.title("Sinal de Saída ao Longo do Tempo")
plt.legend(loc="upper right", bbox_to_anchor=(0.98, 0.99))
plt.grid(True)
plt.show()

#=========== Modelo com RBF =========================
# Função Gaussiana
def rbf(x, center, sigma):
    return np.exp(-((x - center) ** 2) / (2 * sigma ** 2))

# Configurar parâmetros da RBF
n_hidden = 4  # Número de neurônios na camada oculta
centers = np.linspace(min(u), max(u), n_hidden)  # Centros uniformemente espaçados (baseados no sinal de entrada)
sigma = (max(u) - min(u)) / n_hidden  # Largura (sigma) dos núcleos RBF

# Construir a matriz Phi para a RBF
Phi = np.zeros((len(u), n_hidden))  # Matriz de ativação da RBF
for i in range(n_hidden):
    Phi[:, i] = rbf(u, centers[i], sigma)

# Adicionar bias (opcional)
Phi = np.hstack((Phi, np.ones((len(u), 1))))  # Adiciona uma coluna de 1s para o bias

#=========== Treinamento (estimativa de pesos) ==============================
# Ajustar os pesos usando a pseudo-inversa
w = np.linalg.pinv(Phi) @ y

#=========== Teste (estimativa da saída) ====================================
y_est = Phi @ w

#=========== Visualização dos resultados ====================================
plt.figure(figsize=(14, 4))
plt.plot(t, y, "b.", label="Dados Reais")
plt.plot(t, y_est, "--r", label="Ajuste com RBF")
plt.xlabel("Tempo (s)")
plt.ylabel("Sinal de Saída")
plt.legend()
plt.title("Ajuste com Rede RBF (1-4-1)")
plt.grid(True)
plt.show()

# Calcular o erro médio quadrático (EMQ)
EMQ = mean_squared_error(y, y_est)
print(f"Erro Médio Quadrático (EMQ): {EMQ:.4f}")
