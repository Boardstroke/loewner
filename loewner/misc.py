import numpy as np
# Implementação da função de weierstrass




def W(b,N,t,M):
  '''
    Parametros:
    ==========
      a,b floats
        Coeficiente da função de weistrass
      N int
        Parametro para truncar a série,
      t ndarray unidimensional
        Intervalo de tempo em que está definido a função
      M int
        Quantidade de pontos na discretização de pontos

    Retorna
    =======
      y ndarray(1,M)
        Portanto a transposta do array é o função de weistrass no intervalo t
  '''
  y = np.zeros((1,M))
  for n in range(1,N):
    y = y + np.cos(b**n * np.pi * t)*(b**(-n/2))

  return y



