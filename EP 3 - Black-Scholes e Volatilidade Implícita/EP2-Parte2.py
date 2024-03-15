import math
import pandas as pd
import matplotlib.pyplot as plt
#Função que calcula d1:
def calc_d1(t, T, K, St, r, sigma):
    num= math.log(St/K) + (r + ((sigma**2)/2))*(T-t)
    denom= sigma*(math.sqrt(T-t))
    return num/denom

#Função que calcula d2:
def calc_d2(t, T, sigma, d1):
    return d1 - (sigma*(math.sqrt(T-t)))

#Função que calcula o valor da distribuição normal padronizada em um ponto específico "a":
def func_distr(a):
    num= math.exp((-1/2)*(a**2))
    denom= math.sqrt(2*math.pi)
    return num/denom

#Função que calcula a função N(x), através de Simpson, recebe o argumento "x" de N(x) e "n" o número de repetições para o Método de Simpson (default é n=1):
def calc_N(x,n=1):
    h=x/(2*n)
    contador=1

    #soma f(x0) e f(xn)
    somatorio= func_distr(0) + func_distr(x) 
    #soma valores intermediários
    while contador < (2*n):
        if contador%2!=0:
            somatorio+=4*func_distr(contador*h)
        else:
            somatorio+=2*func_distr(contador*h)
        contador+=1
    resultado= (h/3)*somatorio
    return 0.5 + resultado

#Função que calcula Ct:
def calc_Ct(t, T, K, St, r, sigma):
    d1 = calc_d1(t, T, K, St, r, sigma)
    d2 = calc_d2(t, T, sigma, d1)
    parte1= St*calc_N(d1)
    parte2 = K*(math.exp((-1)*r*(T-t)))*calc_N(d2)
    return parte1 - parte2

#Função auxiliar, usada para calcular Newton:
def aux(St, K,  t, T, r, sigma):
    d1= calc_d1(t, T, K, St, r, sigma)
    aux = St*func_distr(d1)*math.sqrt(T-t)
    return aux
# ((1/math.sqrt(2*math.pi))*math.exp((-0.5)*(d1**2))*(math.sqrt(T-t)-d1*(1/sigma)))

#Função que faz o Método de Newton
def newton(Ct_barra, St, K, t, T, r, erro=1e-4, max_iter=100):

    #Estimativa de volatilidade incial
    sigma=1

    for i in range(max_iter):

        #Calcula a diferença Ct-Ct_barra
        dif=calc_Ct(t, T, K, St, r, sigma) - Ct_barra

        #Para o algoritmo quando a diferença é menor ou igual ao erro
        if abs(dif)<= erro:
            break

        #Usa Newton-Raphson para atualizar o valor de sigma, iterativamente
        sigma=sigma-(dif/aux(St, K,  t, T, r, sigma))
    return sigma

#Carregando e tratando os dados das ações da American Express
file_path = 'AXP.xlsx'
df = pd.read_excel(file_path, sheet_name='Plan1')
df = df.drop(df.columns[1], axis=1)
df['t'] = df.index/252

date1 = pd.Timestamp('2024-01-19')
date2 = pd.Timestamp('2023-11-01')
T = (date1 - date2).days
df['T'] = T/252

S_values = column_list = df['S'].tolist()
C_barra_values = column_list = df['Cbarra'].tolist()
t_values = column_list = df['t'].tolist()
T_values = column_list = df['T'].tolist()

r_USA=5.33/100
K_opt=200

Vol_impli = []
for k in range(0,len(S_values)):
    Vol_impli.append(newton(Ct_barra = C_barra_values[k], St = S_values[k], K = K_opt, t = t_values[k], T = T_values[k], r = r_USA))


plt.figure(figsize=(10, 6))
plt.plot(t_values, S_values , label='AXP')
plt.title('Variação da ação')
plt.xlabel('Tempo')
plt.ylabel('Valor de AXP')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(t_values, C_barra_values , label='Call de AXP')
plt.title('Variação da opção')
plt.xlabel('Tempo')
plt.ylabel('Call de AXP')
plt.legend()
plt.grid(True)
plt.show()

# Criar o gráfico
plt.figure(figsize=(10, 6))
plt.plot(t_values, Vol_impli , label='Volatilidade Implícita')
plt.title('Volatilidade implícita ao longo do tempo')
plt.xlabel('Tempo')
plt.ylabel('Volatilidade implícita da opção')
plt.legend()
plt.grid(True)
plt.show()
