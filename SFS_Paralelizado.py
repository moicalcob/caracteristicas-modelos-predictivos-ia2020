import pandas as pandas
import time
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from sklearn.model_selection import cross_val_score
from sklearn import tree
from funcy import join
from sklearn import preprocessing
pandas.set_option('max_colwidth', 800)


def evaluar_soluciones(datos, solucion_actual, nueva_variable, objetivo, n_exp, cv, clf=tree.DecisionTreeClassifier(),
                       scoring="balanced_accuracy"):
    variables = solucion_actual[:]
    variables.append(nueva_variable)
    data_frame = pandas.DataFrame(data=datos)
    X = data_frame[variables]
    y = data_frame[objetivo]

    scores = np.mean(cross_val_score(clf, X, y, cv=cv, scoring=scoring))

    for i in range(n_exp - 1):
        new_scores = np.mean(cross_val_score(clf, X, y, cv=cv, scoring=scoring))
        scores = scores + new_scores

    scores = scores / n_exp

    diccionario_resultado = {}
    diccionario_resultado[nueva_variable] = scores
    return diccionario_resultado


def evaluar_soluciones_eliminando(datos, solucion_actual, variable_a_eliminar, objetivo, n_exp, cv,
                                  clf=tree.DecisionTreeClassifier(), scoring="balanced_accuracy"):
    variables = solucion_actual[:]
    variables.remove(variable_a_eliminar)
    if len(variables) < 1:
        return
    data_frame = pandas.DataFrame(data=datos)
    X = data_frame[variables]
    y = data_frame[objetivo]

    scores = np.mean(cross_val_score(clf, X, y, cv=cv, scoring=scoring))

    for i in range(n_exp - 1):
        new_scores = np.mean(cross_val_score(clf, X, y, cv=cv, scoring=scoring))
        scores = scores + new_scores

    scores = scores / n_exp

    diccionario_resultado = {}
    diccionario_resultado[variable_a_eliminar] = scores
    return diccionario_resultado


def SFS(datos, respuesta, d=0):
    start = time.time()

    diccionario_resultado = {}
    columnas = list(datos.columns)
    columnas.remove(respuesta)
    solucion_actual = []

    k = 0
    d = d if d else len(columnas)

    while (k < d):
        pool = mp.Pool(mp.cpu_count())
        new_resultados = pool.starmap(evaluar_soluciones,
                                      [(datos, solucion_actual, nuevaVariable, respuesta, 15, 10) for nuevaVariable in
                                       columnas])
        pool.close()
        resultado = join(new_resultados)

        variable_escogida = max(resultado, key=resultado.get)
        solucion_actual.append(variable_escogida)
        columnas.remove(variable_escogida)

        k = k + 1

        diccionario_resultado[", ".join(solucion_actual)] = resultado[variable_escogida]

    done = time.time()
    elapsed = done - start
    print("Tiempo empleado: ", elapsed)
    return diccionario_resultado