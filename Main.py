from turtle import forward, right
from matplotlib import markers
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button
import matplotlib as mpl
import math

from adaline import Adaline


class Ventana:
    puntos, clase_deseada = np.array([]), []
    sin_evaluar=np.array([])
    adaline=None
    epoca_actual=0
    epocas_maximas=0
    rango=0.1
    error_minimo=0.1
    rango_inicializado=False
    pesos_inicializados=False
    adaline_entrenado=False
    linea=None
    texto_de_epoca = None
    termino=False
    errores=[]
    total=0
    deseada_deseada=0
    deseada_nodeseada=0
    nodeseada_deseada=0
    nodeseada_nodeseada=0

    def __init__(self):
        #Configuracion inicial de la interfaz grafica.
        mpl.rcParams['toolbar'] = 'None'
        self.fig, (self.grafica, self.grafica_errores) = plt.subplots(2, 1)
        self.fig.canvas.manager.set_window_title('Adaline')
        self.fig.set_size_inches(10, 8, forward=True)
        plt.subplots_adjust(bottom=0.150, top=0.850)
        self.grafica.set_xlim(-1.0,1.0)
        self.grafica.set_ylim(-1.0,1.0)
        self.fig.suptitle("Algoritmo del Adaline")
        self.grafica_errores.set_title("Errores")
        self.grafica_errores.set_xlabel("Epoca")
        self.grafica_errores.set_ylabel("Errores")
    # Acomodo de los botones y cajas de texto
        cordenadas_rango = plt.axes([0.200, 0.9, 0.100, 0.03])
        coordenadas_epcoas = plt.axes([0.440, 0.9, 0.100, 0.03])
        coordenadas_error_deseado = plt.axes([0.720, 0.9, 0.100, 0.03])
        coordenadas_pesos = plt.axes([0.025, 0.05, 0.125, 0.03])
        coordenadas_evaluar = plt.axes([0.160, 0.05, 0.1, 0.03])
        coordenadas_reiniciar = plt.axes([0.270, 0.05, 0.1, 0.03])
        coordenadas_entrenar_adaline = plt.axes([0.680, 0.05, 0.1, 0.03])
        coordenadas_entrenar_perceptron = plt.axes([0.800, 0.05, 0.1, 0.03])
        self.text_box_rango = TextBox(cordenadas_rango, "Rango de aprendizaje:")
        self.text_box_epocas = TextBox(coordenadas_epcoas, "Épocas maximas:")
        self.text_box_error_minimo_deseado = TextBox(coordenadas_error_deseado, "Error mínimo deseado:")
        boton_pesos = Button(coordenadas_pesos, "Inicializar pesos")
        boton_evaluar = Button(coordenadas_evaluar, "Evaluar")
        boton_reiniciar = Button(coordenadas_reiniciar, "Reiniciar")
        boton_entrenar_adaline = Button(coordenadas_entrenar_adaline, "Adaline")
        boton_entrenar_perceptron = Button(coordenadas_entrenar_perceptron, "Perceptron")
        self.text_box_epocas.on_submit(self.validar_epocas)
        self.text_box_rango.on_submit(self.validar_rango)
        self.text_box_error_minimo_deseado.on_submit(self.validar_error_minimo_deseado)
        boton_pesos.on_clicked(self.inicializar_pesos)
        boton_evaluar.on_clicked(self.evaluar)
        boton_reiniciar.on_clicked(self.reiniciar)
        boton_entrenar_adaline.on_clicked(self.entrenar_adaline)
        boton_entrenar_perceptron.on_clicked(self.entrenar_perceptron)
        self.fig.canvas.mpl_connect('button_press_event', self.__onclick)
        plt.show()

    def __onclick(self, event):
        if event.inaxes == self.grafica:
            current_point = [event.xdata, event.ydata]
            if self.adaline_entrenado:
                self.grafica.plot(event.xdata, event.ydata,'ks')
                self.sin_evaluar=np.append(self.sin_evaluar, [event.xdata, event.ydata]).reshape([len(self.sin_evaluar) + 1, 2])
            else:
                self.puntos = np.append(self.puntos, current_point).reshape([len(self.puntos) + 1, 2])
                is_left_click = event.button == 1               
                self.clase_deseada.append(0 if is_left_click else 1)
                self.grafica.plot(event.xdata, event.ydata, 'b.' if is_left_click else 'rx')
            self.fig.canvas.draw()


    def entrenar_adaline(self, event):
        if self.pesos_inicializados and not self.adaline_entrenado:
            error_cuadratico = 1
            self.total=len(self.puntos)
            while error_cuadratico > self.error_minimo and self.epoca_actual < self.adaline.epocas_maximas:   
                self.deseada_deseada=0
                self.deseada_nodeseada=0
                self.nodeseada_deseada=0
                self.nodeseada_nodeseada=0
                self.epoca_actual += 1
                error_cuadratico = 0
                for i, x in enumerate(self.puntos):
                    x = np.insert(x, 0, -1.0)
                    y=self.adaline.f(x)
                    error = self.clase_deseada[i] - y
                    if(self.clase_deseada[i]==1):
                        if(y>.5):#deseada y deseada
                            self.deseada_deseada+=1
                        else:
                            self.deseada_nodeseada+=1
                    else:
                        if(y>.5):#nodeseada y deseada
                            self.nodeseada_deseada+=1
                        else:
                            self.nodeseada_nodeseada+=1
                    derivada = y * ( 1- y)
                    diferencial=2*self.adaline.rango * error * derivada
                    error_cuadratico+=error**2
                    self.adaline.pesos = self.adaline.pesos + np.multiply((diferencial ), x)
                    self.graficar_linea()
                self.errores.append(error_cuadratico)
                self.graficar_errores()
            
            if(error_cuadratico < self.error_minimo):
                self.termino=True
            self.grafica.text(0, -1.250,'Resultado = ' + ('Error minimo obtenido' if self.termino else 'No converge'),fontsize=16) #cambiar validacion
            self.texto_de_epoca.set_text("Épocas: %s" % self.epoca_actual)
            plt.pause(0.1)
            self.adaline_entrenado = True
            print("Matriz de confusion")
            print("N="+str(self.total))
            print("\t\tObtenida si\tObtenida no")
            print("Deseada si\t"+str(self.deseada_deseada)+"\t\t"+str(self.deseada_nodeseada))
            print("Deseada No\t"+str(self.nodeseada_deseada)+"\t\t"+str(self.nodeseada_nodeseada))
            print("\t\t"+str(self.deseada_deseada+self.nodeseada_deseada)+"\t\t"+str(self.deseada_nodeseada+self.nodeseada_nodeseada))
    
    def graficar_errores(self):
        x=self.errores
        y = range(len(x))
        self.grafica_errores.clear()
        self.grafica_errores.plot(y,x)
        plt.pause(0.3)
    def evaluar(self,event):
        if(self.adaline_entrenado and len(self.sin_evaluar)>0):
            self.grafica.clear() 
            self.barrido()
            self.grafica.set_xlim(-1.0,1.0)
            self.grafica.set_ylim(-1.0,1.0)
            
            for j,k in enumerate(self.puntos):
                self.grafica.plot(k[0], k[1], 'r.' if not self.clase_deseada[j] else 'bx')            
            
            self.grafica.plot(self.linea.get_xdata(), self.linea.get_ydata(), 'y-')
            self.grafica.text(0.8, 0.9,'Época: %s' % self.epoca_actual, fontsize=10)
            
            for  i,x in enumerate(self.sin_evaluar):
                x = np.insert(x, 0, -1.0)
                self.grafica.plot(x[1], x[2], 'gx' if self.adaline.f(x)>.5 else 'g.')
           
            plt.pause(0.1)


    def inicializar_pesos(self, event):
        if self.rango_inicializado and self.epocas_maximas>0 and len(self.puntos)>0 and not self.adaline_entrenado:
            self.adaline = Adaline(self.rango, self.epocas_maximas, [-1.0,1])
            self.adaline.inicializar_pesos()
            self.pesos_inicializados = True
            self.graficar_linea()
        

    def graficar_linea(self):
        x1 = np.array([self.puntos[:, 0].min() - 2, self.puntos[:, 0].max() + 2])
        m = -self.adaline.pesos[1] / self.adaline.pesos[2]
        c = self.adaline.pesos[0] / self.adaline.pesos[2]
        x2 = m * x1 + c
        
        if not self.linea:
            self.linea, = self.grafica.plot(x1, x2, 'y-')
            self.texto_de_epoca = self.grafica.text(0.8, 0.9, 'Época: %s' % self.epoca_actual, fontsize=10)
        else:
            self.linea.set_xdata(x1)
            self.linea.set_ydata(x2)
            self.texto_de_epoca.set_text('Época: %s' % self.epoca_actual)
        self.fig.canvas.draw()
        plt.pause(0.1)


    def validar_rango(self, expression):
        try:
            r=float(expression)
            if(r>0 and r<1):
                self.rango =float(expression)
            else:
                self.rango =0.1    
        except ValueError:
            self.rango =0.1
        finally:
            self.text_box_rango.set_val(self.rango)
            self.rango_inicializado=True


    def validar_epocas(self, expression):
        try:
            self.epocas_maximas =int(expression)
        except ValueError:
            self.epocas_maximas =50
        finally:
            self.text_box_epocas.set_val(self.epocas_maximas)


    def validar_error_minimo_deseado(self,expression):
        try:
            r=float(expression)
            if(r>0 and r<1):
                self.error_minimo =float(expression)
            else:
                self.error_minimo =0.1    
        except ValueError:
            self.error_minimo =0.1
        finally:
            self.text_box_error_minimo_deseado.set_val(self.error_minimo)

        
    def reiniciar(self, event):
        self.puntos, self.clase_deseada = np.array([]), []
        self.sin_evaluar=np.array([])
        self.adaline=None
        self.epoca_actual=0
        self.epocas_maximas=0
        self.rango=0.1
        self.error_minimo=0.1
        self.rango_inicializado=False
        self.pesos_inicializados=False
        self.adaline_entrenado=False
        self.linea=None
        self.texto_de_epoca = None
        self.termino=False
        self.grafica.clear()
        self.grafica.set_xlim(-1.0,1.0)
        self.grafica.set_ylim(-1.0,1.0)
        self.text_box_rango.set_val('')
        self.text_box_epocas.set_val('')
        self.text_box_error_minimo_deseado.set_val('')
        self.errores=[]
        self.grafica_errores.clear()


    def barrido(self):
        x=-1
        y=0
        while(y>=-1):
            x=-1
            while x<=1:
                p=[x,y]
                j = np.insert(p, 0, -1.0)
                alpha=self.adaline.f(j)
                if (alpha>.5):
                    self.grafica.plot(x,y, color=(1,0,0,alpha),marker='.' )
                else:
                    self.grafica.plot(x,y, color=(0,0,1,alpha),marker='.' )
                
                x+=.01
            y-=.01
            print(y)
            
    def entrenar_perceptron(self):
        pass

if __name__ == '__main__':
    Ventana()