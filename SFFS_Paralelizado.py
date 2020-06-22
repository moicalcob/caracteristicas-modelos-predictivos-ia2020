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


def SFFS(datos, respuesta):
    start = time.time()
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

        # Actualizo el listado de columnas que tengo que evaluar
        columnas_a_evaluar = [x for x in columnas if (x not in añadidos and x not in soluciones_actual)]

        if columnas_a_evaluar == []:
            break

        # Calculamos el nuevo resultado óptimo
        pool = mp.Pool(mp.cpu_count())
        new_resultados = pool.starmap(evaluar_soluciones,
                                      [(datos, soluciones_actual, nuevaVariable, respuesta, 15, 10) for nuevaVariable in
                                       columnas_a_evaluar])
        pool.close()
        # Buscamos la variable más óptima
        resultado = join(new_resultados)
        variable_escogida = max(resultado, key=resultado.get)
        score_resultado = resultado[variable_escogida]
        # Añadimos la variable
        soluciones_actual.append(variable_escogida)
        añadidos.append(variable_escogida)

        # Pasamos a eliminar variables para comprobar si tengo mejores resultados
        if len(soluciones_actual) > 1:
            variables_a_eliminar = [x for x in soluciones_actual if x not in eliminados]

            pool = mp.Pool(2)
            new_resultados = pool.starmap(evaluar_soluciones_eliminando,
                                          [(datos, soluciones_actual, variable_a_eliminar, respuesta, 15, 10) for
                                           variable_a_eliminar in variables_a_eliminar])
            pool.close()
            resultado = join(new_resultados)
            variable_a_eliminar = max(resultado, key=resultado.get)

            score_resultado_eliminado = resultado[variable_a_eliminar]

            if score_resultado < score_resultado_eliminado:
                soluciones_actual.remove(variable_a_eliminar)
                eliminados.append(variable_a_eliminar)
                score_resultado = score_resultado_eliminado
                k = 0

        if len(columnas) == len(datos.columns) - 1:
            k = k + 1

        if len(añadidos) < len(columnas):
            clave = ', '.join(soluciones_actual)
            diccionario_resultado[clave] = score_resultado

    done = time.time()
    elapsed = done - start
    print("Tiempo empleado: ", elapsed)
    return diccionario_resultado