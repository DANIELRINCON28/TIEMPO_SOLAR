"""
ml_engine.py — Motor de Machine Learning para SolarMotion Tracker & Modeler.

Contiene funciones para entrenar modelos de regresión sobre datos empíricos
de movimiento solar e intensidad lumínica, retornando predicciones, métricas
y coeficientes donde aplique.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score


# ---------------------------------------------------------------------------
# Utilidades internas
# ---------------------------------------------------------------------------

def _reshape(arr):
    """Convierte un array 1-D a columna 2-D para scikit-learn."""
    a = np.asarray(arr, dtype=float)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    return a


def _metrics(y_true, y_pred):
    """Calcula MSE, RMSE y R² entre valores reales y predichos."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {"MSE": round(mse, 6), "RMSE": round(rmse, 6), "R2": round(r2, 6)}


def _smooth_x(X, n_points=300):
    """Genera un rango continuo de X para trazar curvas suaves."""
    x_min, x_max = float(np.min(X)), float(np.max(X))
    margin = (x_max - x_min) * 0.05
    return np.linspace(x_min - margin, x_max + margin, n_points).reshape(-1, 1)


# ---------------------------------------------------------------------------
# 1. Regresión Lineal
# ---------------------------------------------------------------------------

def fit_linear(X, Y):
    """
    Entrena una regresión lineal simple.

    Returns
    -------
    dict con claves: model, y_pred, metrics, x_smooth, y_smooth, equation
    """
    X2 = _reshape(X)
    Y2 = np.asarray(Y, dtype=float)

    model = LinearRegression()
    model.fit(X2, Y2)

    y_pred = model.predict(X2)
    metrics = _metrics(Y2, y_pred)

    x_s = _smooth_x(X2)
    y_s = model.predict(x_s)

    coef = model.coef_[0]
    intercept = model.intercept_
    equation = f"y = {coef:.4f}·x + {intercept:.4f}"

    return {
        "model": model,
        "y_pred": y_pred,
        "metrics": metrics,
        "x_smooth": x_s.flatten(),
        "y_smooth": y_s,
        "equation": equation,
        "coefficients": [coef, intercept],
    }


# ---------------------------------------------------------------------------
# 2. Regresión Polinomial
# ---------------------------------------------------------------------------

def fit_polynomial(X, Y, degree=2):
    """
    Regresión polinomial de grado *degree*.

    Returns
    -------
    dict con claves: model, y_pred, metrics, x_smooth, y_smooth, equation,
                     poly_coefficients (mayor grado primero, compatible con np.polyval)
    """
    X2 = _reshape(X)
    Y2 = np.asarray(Y, dtype=float)

    if len(Y2) <= degree:
        raise ValueError(
            f"Se necesitan al menos {degree + 1} puntos para un polinomio de grado {degree}. "
            f"Solo se proporcionaron {len(Y2)}."
        )

    poly = PolynomialFeatures(degree=degree, include_bias=False)
    lr = LinearRegression()
    pipe = Pipeline([("poly", poly), ("lr", lr)])
    pipe.fit(X2, Y2)

    y_pred = pipe.predict(X2)
    metrics = _metrics(Y2, y_pred)

    x_s = _smooth_x(X2)
    y_s = pipe.predict(x_s)

    # Coeficientes en orden decreciente para np.polyval / np.polyder
    raw_coefs = lr.coef_           # [c1, c2, ..., cn]  (ascendente)
    intercept = lr.intercept_
    # Construir array de mayor a menor grado
    poly_coefficients = np.zeros(degree + 1)
    poly_coefficients[-1] = intercept
    for i, c in enumerate(raw_coefs):
        poly_coefficients[degree - 1 - i] = c
    # Re-armar correctamente: usar np.polyfit que es más fiable
    poly_coefficients = np.polyfit(np.asarray(X, dtype=float).flatten(), Y2, degree)

    # Ecuación legible
    terms = []
    for i, c in enumerate(poly_coefficients):
        power = degree - i
        c_round = round(c, 4)
        if c_round == 0:
            continue
        sign = " + " if c_round > 0 and terms else (" - " if c_round < 0 and terms else ("" if c_round >= 0 else "-"))
        abs_c = abs(c_round)
        if power == 0:
            terms.append(f"{sign}{abs_c}")
        elif power == 1:
            terms.append(f"{sign}{abs_c}·x")
        else:
            terms.append(f"{sign}{abs_c}·x^{power}")
    equation = "y = " + "".join(terms) if terms else "y = 0"

    return {
        "model": pipe,
        "y_pred": y_pred,
        "metrics": metrics,
        "x_smooth": x_s.flatten(),
        "y_smooth": y_s,
        "equation": equation,
        "poly_coefficients": poly_coefficients,
    }


# ---------------------------------------------------------------------------
# 3. Árbol de Decisión
# ---------------------------------------------------------------------------

def fit_decision_tree(X, Y, max_depth=5):
    X2 = _reshape(X)
    Y2 = np.asarray(Y, dtype=float)

    model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
    model.fit(X2, Y2)

    y_pred = model.predict(X2)
    metrics = _metrics(Y2, y_pred)

    x_s = _smooth_x(X2, n_points=500)
    y_s = model.predict(x_s)

    return {
        "model": model,
        "y_pred": y_pred,
        "metrics": metrics,
        "x_smooth": x_s.flatten(),
        "y_smooth": y_s,
        "equation": f"DecisionTree(max_depth={max_depth})",
    }


# ---------------------------------------------------------------------------
# 4. SVR (Support Vector Regression)
# ---------------------------------------------------------------------------

def fit_svr(X, Y, kernel="rbf", C=1.0, epsilon=0.1):
    X2 = _reshape(X)
    Y2 = np.asarray(Y, dtype=float)

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_x.fit_transform(X2)
    Y_scaled = scaler_y.fit_transform(Y2.reshape(-1, 1)).ravel()

    model = SVR(kernel=kernel, C=C, epsilon=epsilon)
    model.fit(X_scaled, Y_scaled)

    y_pred_scaled = model.predict(X_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    metrics = _metrics(Y2, y_pred)

    x_s = _smooth_x(X2)
    x_s_scaled = scaler_x.transform(x_s)
    y_s_scaled = model.predict(x_s_scaled)
    y_s = scaler_y.inverse_transform(y_s_scaled.reshape(-1, 1)).ravel()

    return {
        "model": model,
        "y_pred": y_pred,
        "metrics": metrics,
        "x_smooth": x_s.flatten(),
        "y_smooth": y_s,
        "equation": f"SVR(kernel={kernel}, C={C}, ε={epsilon})",
        "_scaler_x": scaler_x,
        "_scaler_y": scaler_y,
    }


# ---------------------------------------------------------------------------
# 5. KNN
# ---------------------------------------------------------------------------

def fit_knn(X, Y, n_neighbors=5):
    X2 = _reshape(X)
    Y2 = np.asarray(Y, dtype=float)

    k = min(n_neighbors, len(Y2))
    model = KNeighborsRegressor(n_neighbors=k)
    model.fit(X2, Y2)

    y_pred = model.predict(X2)
    metrics = _metrics(Y2, y_pred)

    x_s = _smooth_x(X2, n_points=500)
    y_s = model.predict(x_s)

    return {
        "model": model,
        "y_pred": y_pred,
        "metrics": metrics,
        "x_smooth": x_s.flatten(),
        "y_smooth": y_s,
        "equation": f"KNN(k={k})",
    }


# ---------------------------------------------------------------------------
# 6. Random Forest
# ---------------------------------------------------------------------------

def fit_random_forest(X, Y, n_estimators=100, max_depth=5):
    X2 = _reshape(X)
    Y2 = np.asarray(Y, dtype=float)

    model = RandomForestRegressor(
        n_estimators=n_estimators, max_depth=max_depth, random_state=42
    )
    model.fit(X2, Y2)

    y_pred = model.predict(X2)
    metrics = _metrics(Y2, y_pred)

    x_s = _smooth_x(X2, n_points=500)
    y_s = model.predict(x_s)

    return {
        "model": model,
        "y_pred": y_pred,
        "metrics": metrics,
        "x_smooth": x_s.flatten(),
        "y_smooth": y_s,
        "equation": f"RandomForest(n={n_estimators}, depth={max_depth})",
    }


# ---------------------------------------------------------------------------
# 7. Derivada polinomial (velocidad)
# ---------------------------------------------------------------------------

def compute_derivative(poly_coefficients, x_values):
    """
    Calcula la primera derivada de un polinomio.

    Parameters
    ----------
    poly_coefficients : array-like
        Coeficientes del polinomio en orden decreciente (como np.polyfit).
    x_values : array-like
        Puntos donde evaluar la derivada.

    Returns
    -------
    dict con claves: derivative_coefficients, y_derivative, equation
    """
    deriv_coefs = np.polyder(poly_coefficients)
    x_arr = np.asarray(x_values, dtype=float)
    y_deriv = np.polyval(deriv_coefs, x_arr)

    degree = len(deriv_coefs) - 1
    terms = []
    for i, c in enumerate(deriv_coefs):
        power = degree - i
        c_round = round(c, 4)
        if c_round == 0:
            continue
        sign = " + " if c_round > 0 and terms else (" - " if c_round < 0 and terms else ("" if c_round >= 0 else "-"))
        abs_c = abs(c_round)
        if power == 0:
            terms.append(f"{sign}{abs_c}")
        elif power == 1:
            terms.append(f"{sign}{abs_c}·x")
        else:
            terms.append(f"{sign}{abs_c}·x^{power}")
    eq = "v(x) = " + "".join(terms) if terms else "v(x) = 0"

    return {
        "derivative_coefficients": deriv_coefs,
        "y_derivative": y_deriv,
        "equation": eq,
    }


# ---------------------------------------------------------------------------
# Dispatcher — función de conveniencia
# ---------------------------------------------------------------------------

AVAILABLE_MODELS = [
    "Regresión Lineal",
    "Regresión Polinomial",
    "Árbol de Decisión",
    "SVR",
    "KNN",
    "Random Forest",
]


def fit_model(name: str, X, Y, **kwargs):
    """
    Entrena el modelo indicado por *name* con los datos X, Y.

    Parameters
    ----------
    name : str
        Nombre del modelo (ver AVAILABLE_MODELS).
    X, Y : array-like
        Datos de entrada.
    **kwargs
        Hiperparámetros específicos del modelo.

    Returns
    -------
    dict con resultados del entrenamiento.
    """
    dispatch = {
        "Regresión Lineal": fit_linear,
        "Regresión Polinomial": fit_polynomial,
        "Árbol de Decisión": fit_decision_tree,
        "SVR": fit_svr,
        "KNN": fit_knn,
        "Random Forest": fit_random_forest,
    }
    func = dispatch.get(name)
    if func is None:
        raise ValueError(f"Modelo '{name}' no soportado. Opciones: {AVAILABLE_MODELS}")
    return func(X, Y, **kwargs)
