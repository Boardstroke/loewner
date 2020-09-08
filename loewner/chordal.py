import numpy as np
import numexpr as ne

'''
Método para solucionar a equação de loewner em sua versão cordal.
Isso é equação diferencial definida no plano superior
'''

def vslit_zip(z,dt,du):
  '''
    Essa função é a inversa da solução da equação de Loewner com a `driving function`

    U(t) = du, para 0 <= t <= dt

    Leva a origem para o ponto w = du + 2j * dt**0.5

    Parametros
    ----------
    z : ndarray de números complexos
        Input array.

    dt: float
    du: float
        Vertical slit parametros.

    Returns
    -------
    w : ndarray de números complexos
        Possuem mesma shape do vetor z.

  '''
  return ne.evaluate("1j * sqrt(4 * dt - z ** 2) + du");


def trace(t, u):
    '''


    Computa o traço discretizado da equação de loewner associada a `driving function` u(t) em um tempos discretizados. O traço é definido como a curva z(t), tal que f(t, u(t)) onde f(t,w) é a inversa da função g(t,w), que a solução da equação de loewner no plano superior.


    dg          2
    -- = ----------------
    dt    g(t, w) - u(t)

    Faz uso do algoritmo de zipper com a discretização vertical (função acima)

    Referência
    ---------
    Kennedy, Tom. "Numerical computations for the Schramm-Loewner evolution."
    Journal of Statistical Physics 137.5-6 (2009): 839-856.

    Parametros
    ----------
    t: ndarray unidimensional de floats
        Com instantes de tempos que vamos retirar as amostras da driving function

    u: ndarray unidimensional de floats
        Nossa driving function
    Returns
    -------
    z: ndarray unidimensional de números complexos
        Pontos do traço da função definda no intervalo discretizado recebido.
    '''
    n = len(t)
    z = np.zeros(n, dtype=np.complex128)
    for i in range(n - 1, 0, -1):
        dt = t[i] - t[i - 1]
        du = u[i] - u[i - 1]
        z[i:] = vslit_zip(z[i:], dt, du)
    return z
