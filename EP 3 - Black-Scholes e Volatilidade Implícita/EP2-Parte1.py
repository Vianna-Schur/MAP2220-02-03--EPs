import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
#f.d.p da Normal
def Normal_fdp(x):
    
    return 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * x**2)

#Implementação do Simpson
def simpson_integral(a, b, n, func):
    if n % 2:
        raise ValueError("n precisa ser par")
    
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = func(x)
    
    S = y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2])
    return S * h / 3

#Cálculo de N(x)
def N(x, n_intervals=10000):
    
    return 0.5 + simpson_integral(0, x, n_intervals, Normal_fdp)


def calculate_d1(S_t, K, r, sigma, T, t):
    numerator = np.log(S_t / K) + (r + sigma**2 / 2) * (T - t)
    denominator = sigma * np.sqrt(T - t)
    return numerator / denominator

def calculate_d2(d1, sigma, T, t):
    return d1 - sigma * np.sqrt(T - t)

# Cálculo do preço pela fórmula de Black-Scholes
def BS_call_price(S_t, K, r, sigma, T, t, d1, d2):
    call_price = S_t * N(d1) - K * np.exp(-r * (T - t)) * N(d2)
    return call_price

# Parâmetros de teste
S_t_teste = 100   # Preço atual
K_teste = 100     # Preço de exercício/Strike
r_teste = 0.1175   # Taxa livre de risco (CDI/Selic no caso Brasileiro)
sigma_teste = 1.0 # Volatilidade teórica
T_teste = 5/252       # Tempo para expiração
t_teste = 0/252       # Tempo atual

D1 = calculate_d1(S_t_teste, K_teste, r_teste, sigma_teste, T_teste, t_teste)
D2 = calculate_d2(D1, sigma_teste, T_teste, t_teste)
# Cálculo da opção de compra (call)
call_price = BS_call_price(S_t_teste, K_teste, r_teste, sigma_teste, T_teste, t_teste, D1, D2)

print(call_price)

# Gerar um intervalo de valores para S_t
S_t_values = np.linspace(70, 130, 100)  

# Calcular o preço da opção de compra para cada valor de S_t
call_prices = []
for S_t in S_t_values:
    D1 = calculate_d1(S_t, K_teste, r_teste, sigma_teste, T_teste, t_teste)
    D2 = calculate_d2(D1, sigma_teste, T_teste, t_teste)
    call_price = BS_call_price(S_t, K_teste, r_teste, sigma_teste, T_teste, t_teste, D1, D2)
    call_prices.append(call_price)

# Criar o gráfico
plt.figure(figsize=(10, 6))
plt.plot(S_t_values, call_prices, label='Preço da Opção de Compra (Call)')
plt.title('Variação no Valor da Opção VS Preço Atual (S_t)')
plt.xlabel('Preço Atual do Ativo Subjacente (S_t)')
plt.axvline(x=K_teste, color='red', linestyle='--', label='Preço de Exercício (Strike) K')
plt.ylabel('Preço da Opção de Compra (Call)')
plt.legend()
plt.grid(True)
plt.show()


