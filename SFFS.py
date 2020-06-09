import pandas as pandas;
from sklearn.model_selection import cross_val_score;
from sklearn import tree;
from sklearn import preprocessing;
import numpy as np;
pandas.set_option('max_colwidth', 800)

def evaluar_soluciones(datos, variables, objetivo, n_exp, cv):
 data_frame = pandas.DataFrame(data=datos)
 X = data_frame[variables]
 y = data_frame[objetivo]
 clf = tree.DecisionTreeClassifier()

 scores = cross_val_score(clf, X, y, cv=cv, scoring="balanced_accuracy")

 for i in range(n_exp - 1):
  new_scores = cross_val_score(clf, X, y, cv=cv, scoring="balanced_accuracy")
  scores = scores + new_scores

 scores = scores / n_exp
 return np.mean(scores)


def SFFS(datos, respuesta):
 diccionario_resultado = {}
 soluciones_actual = []
 añadidos = []
 eliminados = []
 columnas = list(datos.columns)
 k = 0

 # Compruebo que la variable a predecir no esté en mi conjunto de variables a evaluar
 if respuesta in columnas:
  columnas.remove(respuesta)

 while (k < 10):
  resultado = []
  score_resultado = 0
  score_resultado_eliminado = 0
  resultado_eliminado = []
  eliminado = ''

  for i in range(len(columnas)):

   # Compruebo que la nueva variable a evaluar no haya sido ya evaluada o este en añadidos
   if columnas[i] not in soluciones_actual and columnas[i] not in añadidos:
    solucionTemporal = list(soluciones_actual)
    solucionTemporal.append(columnas[i])
    new_resultado = evaluar_soluciones(datos, solucionTemporal, respuesta, 15, 10)

    # Si el resultado es favorable, actualizo el resultado final
    if new_resultado > score_resultado:
     resultado = solucionTemporal
     score_resultado = new_resultado

  if len(resultado) > 0:
   soluciones_actual.append(resultado[len(resultado) - 1])
   añadidos.append(resultado[len(resultado) - 1])

  score_resultado_eliminado = score_resultado

  if len(soluciones_actual) > 1:
   for i in range(len(soluciones_actual)):

    # Compruebo que la variable a evaluar no este en eliminados
    if soluciones_actual[i] not in eliminados:
     solucionTemporal = list(soluciones_actual)
     solucionTemporal.remove(soluciones_actual[i])
     new_resultado = evaluar_soluciones(datos, solucionTemporal, respuesta, 15, 10)

     # Si el resultado es favorable, actualizo el resultado para eliminar la variable actual que
     # ha sido quitada de la solución actual
     if new_resultado > score_resultado_eliminado:
      resultado_eliminado = solucionTemporal
      score_resultado_eliminado = new_resultado
      eliminado = soluciones_actual[i]

   if score_resultado < score_resultado_eliminado:
    soluciones_actual = resultado_eliminado
    eliminados.append(eliminado)
    k = 0

  k = k + 1

  if len(añadidos) < len(columnas):
   print(soluciones_actual)
   print(score_resultado_eliminado)
   clave = ', '.join(soluciones_actual)
   diccionario_resultado[clave] = score_resultado_eliminado

 return diccionario_resultado