import numpy as np
import matplotlib.pyplot as plt
S0 = 2
r = 0.045
r0 = 0.02
S_bar = 5



# Função de retorno em renda fixa
def Sf(t):
    return S0 * np.exp(r * t)

# Função de retorno logístico
def Srs(t):
    return S_bar / (1 + ((S_bar / S0) - 1) * np.exp(-r0 * S_bar * t))

# Função para encontrar a diferença entre os retornos dos dois investimentos
def Diferenca(t):
    return Sf(t) - Srs(t)

# Método da dicotomia para encontrar o ponto de interseção
def dicotomia(f, a, b, tol):
    if f(a) * f(b) >= 0:
        print("O método da dicotomia falhou. f(a) e f(b) devem ter sinais opostos")
        return None

    c = (a + b) / 2.0  # Ponto médio inicial
    while (b - a) / 2.0 > tol:
        if f(c) == 0:
            return c  # A raiz exata foi encontrada
        elif f(a) * f(c) < 0:
            b = c
        else:
            a = c
        c = (a + b) / 2.0

    return c  # A raiz aproximada

# Parâmetros do modelo
S0 = 2
r = 0.045
r0 = 0.02
S_bar = 5

# Aplicar o método da dicotomia
t_intersection = dicotomia(Diferenca, 0.01, 100, 1e-6)
print(f"O ponto de interseção é aproximadamente em t = {t_intersection:.2f} unidades de tempo.")

# Gerar valores de t e calcular os retornos correspondentes para cada investimento
t_values = np.linspace(0, t_intersection+8, 400)
Sf_values = Sf(t_values)
Srs_values = Srs(t_values)

# Calcular a diferença entre os dois investimentos
difference = Sf_values - Srs_values

# Criar gráficos
#plt.figure(figsize=(14, 7))

# Gráfico da evolução dos investimentos
#plt.subplot(1, 2, 1)
plt.plot(t_values, Sf_values, label='Renda Fixa', color='blue')
plt.plot(t_values, Srs_values, label='Retorno Logístico', color='orange')
plt.axvline(x=t_intersection, color='red', linestyle='--', label=f'Interseção em t={t_intersection:.2f}')
plt.title('Evolução dos Investimentos')
plt.xlabel('Tempo')
plt.ylabel('Retorno Acumulado')
plt.legend()

plt.show()

# Gráfico da diferença entre os dois investimentos
#plt.subplot(1, 2, 2)
plt.plot(t_values, difference, label='Diferença', color='green')
plt.axhline(y=0, color='black', linestyle='--')
plt.axvline(x=t_intersection, color='red', linestyle='--', label=f'Interseção em t={t_intersection:.2f}')
plt.title('Diferença entre os Investidores')
plt.xlabel('Tempo')
plt.ylabel('Diferença no Valor')
plt.legend()

plt.tight_layout()
plt.show()