import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import quad
from scipy.optimize import root_scalar
import pandas as pd
import seaborn as sns
from mpmath import mp
mp.dps = 50
num_points = 71
# Carrega os dados
df = pd.read_excel('C:\EPs - Fundamentos de Análise Numérica\EP1 - População e Petróleo\PopMundial.xlsx', sheet_name='Planilha1')

# Calcula a variação da população ano a ano
df['Variation of the Population'] = df['Population'].diff()

# Primeira variação é nula, mas é descartada a primeira linha da tabela
df['Variation of the Population'].iloc[0] = 0
df = df.drop(df.index[0])
df = df.reset_index(drop=True)

x_data = [mp.mpf(val) for val in df['Population']]
y_data = [mp.mpf(val) for val in df['Variation of the Population']]

# Como temos dados com valores altos, para evitar problemas de arrendondamento catastrófico etc, normalizamos

# Reescala das variáveis
x_max = max(x_data)
y_max = max(y_data)

x_scaled = [val/x_max for val in x_data]
y_scaled = [val/y_max for val in y_data]

# Construindo a matriz M
M11 = mp.fsum(xi**2 * xj**2 for xi, xj in zip(x_scaled, x_scaled))
M12 = mp.fsum(xi**2 * xj for xi, xj in zip(x_scaled, x_scaled))

M21 = mp.fsum(xi * xj**2 for xi, xj in zip(x_scaled, x_scaled))
M22 = mp.fsum(xi * xj for xi, xj in zip(x_scaled, x_scaled))



M = mp.matrix([[M11, M12],
              [M21, M22]])

print(M)

# Construindo o vetor de termos independentes
v1 = mp.fsum(xi**2 * yi for xi, yi in zip(x_scaled, y_scaled))
v2 = mp.fsum(xi * yi for xi, yi in zip(x_scaled, y_scaled))

v = mp.matrix([v1, v2])

beta = M**-1 * v

a_scaled, b_scaled = beta


#Cálculo de parâmetros
Alpha0 = -(a_scaled*y_max)/(x_max**2)
m = (b_scaled*y_max) /x_max *1/Alpha0
print(f"a: {a_scaled}")
print(f"b: {b_scaled}")

print(f"Alpha0: {Alpha0}")
print(f"M: {m}")



# Montando os pontos para checar o ajuste do MMQ
x_fit_mpmath = [min(x_scaled) + mp.mpf(i) * (max(x_scaled) - min(x_scaled)) / (num_points - 1) for i in range(num_points)]
y_fit_mpmath = [a_scaled * xi**2 + b_scaled * xi for xi in x_fit_mpmath]

# Conversão de volta pra float para poder fazer o gráfico
x_fit_float = [float(xi) for xi in x_fit_mpmath]
y_fit_float = [float(yi) for yi in y_fit_mpmath]

# Gráfico utilizando Seaborn
sns.set(style="ticks")
plt.figure(figsize=(10, 6))
sns.scatterplot(x=x_scaled, y=y_scaled, label='Data')
plt.plot(x_fit_float, y_fit_float, color='red')
plt.tick_params(axis='both', which='major', labelsize=14)
plt.title(r'Taxa de variação da População/Máx VS População/Máx',fontsize=16)
plt.xlabel(r'População/Máx',fontsize=16)
plt.ylabel(r'Variação da População/Máx',fontsize=16)
plt.legend()
plt.show()

#Conversão de volta pra float e definição de ano inicial
P0 = df[df['Year'] == 1951]['Population'].values[0]
t0 = 1951
Alpha0_float = float(Alpha0)
m_float = float(m)


# Define P(t)
def P(t):
    return m_float / (1 + (m_float - P0) / P0 * np.exp(-Alpha0_float * m_float * (t - t0)))

# Gera valores pro gráfico de P(t)
t_values = np.linspace(1831, 2320, 500)
P_values = P(t_values)

sns.scatterplot(x='Year', y='Population', data=df, label='Dados WB')

# Gráfico de P(t)
plt.plot(t_values, P_values, color='green', label=f'Modelo Logístico Ajustado')

plt.title('População vs Ano; Dados Experimentais e Modelo')
plt.xlabel('Ano')
plt.ylabel('População')
plt.legend()
plt.show()

P1 = df[df['Year'] == 2021]['Population'].values[0]
t1 = 2021
capita_Americano = 20.84

# Gera valores para o gráfico
T_values = np.linspace(2021, 2050, 500)

def integrand(t):
    return capita_Americano * m_float / (1 + (m_float - P1) / P1 * np.exp(-Alpha0_float * m_float * (t - t1)))

# Calcula a integral
integral_values = [quad(integrand, 2021, T)[0] for T in T_values]

# Faz o gráfico da integral até T e bota a linha das reservas totais
plt.figure(figsize=(10, 6))
plt.plot(T_values, integral_values, label='Consumo total de Petróleo (Barris)')
plt.axhline(1853849 * 1000000, color='r', linestyle='--', label='Reservas totais contabilizadas')
plt.xlabel('T')
plt.ylabel(f'Barris de Petróleo Consumidos')
plt.legend()
plt.show()