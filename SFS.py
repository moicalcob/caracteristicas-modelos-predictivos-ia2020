import pandas as pandas;
import time;
from sklearn.model_selection import cross_val_score;
from sklearn import tree;
from sklearn import preprocessing;
import numpy as np;
import matplotlib.pyplot as plt
pandas.set_option('max_colwidth', 800)


def evaluar_soluciones(datos, variables, objetivo, n_exp, cv, clf=tree.DecisionTreeClassifier(),
                       scoring="balanced_accuracy"):
    data_frame = pandas.DataFrame(data=datos)
    X = data_frame[variables]
    y = data_frame[objetivo]

    scores = np.mean(cross_val_score(clf, X, y, scoring=scoring, cv=cv, n_jobs=-1))

    for i in range(n_exp - 1):
        new_scores = np.mean(cross_val_score(clf, X, y, scoring=scoring, cv=cv, n_jobs=-1))
        scores = scores + new_scores

    scores = scores / n_exp
    return scores


def SFS(datos, respuesta, d=0):
    start = time.time()

    diccionario_resultado = {}
    soluciones_actual = []
    columnas = list(datos.columns)

    # Compruebo que la variable a predecir no esté en mi conjunto de variables a evaluar
    if respuesta in columnas:
        columnas.remove(respuesta)

    k = 0
    d = d if d else len(columnas)

    while (k < d):
        resultado = []
        score_resultado = 0

        for i in range(len(columnas)):

            # Compruebo que la nueva variable a evaluar no haya sido ya evaluada
            if columnas[i] not in soluciones_actual:
                solucionTemporal = list(soluciones_actual)
                solucionTemporal.append(columnas[i])
                new_resultado = evaluar_soluciones(datos, solucionTemporal, respuesta, 15, 10)

                # Si el resultado es favorable, actualizo el resultado final
                if new_resultado > score_resultado:
                    resultado = solucionTemporal
                    score_resultado = new_resultado

        soluciones_actual.append(resultado[len(resultado) - 1])
        clave = ', '.join(soluciones_actual)
        diccionario_resultado[clave] = score_resultado
        k = k + 1

    done = time.time()
    elapsed = done - start
    print("Tiempo empleado: ", elapsed)

    return diccionario_resultado