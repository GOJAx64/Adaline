import random
import numpy as np


class Perceptron:
    def __init__(self, rango, epocas_maximas, rango_de_normalizacion=None, N=2):
        if rango_de_normalizacion is None:
            rango_de_normalizacion = [-1.0, 1.0]
        self.rango_de_normalizacion = rango_de_normalizacion
        self.rango = rango
        self.N = N
        self.epocas_maximas = epocas_maximas
        self.errores = []
        self.pesos = []

    def inicializar_pesos(self):
        self.pesos = []
        for i in range(self.N+1):               
            self.pesos.append(random.uniform(self.rango_de_normalizacion[0], self.rango_de_normalizacion[1]))
        self.pesos = np.array(self.pesos)

    def pw(self, x):
        return 1 if np.dot(x, self.pesos) >= 0 else 0