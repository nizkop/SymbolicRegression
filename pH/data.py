import numpy as np
import pandas as pd
from gplearn.functions import make_function
from gplearn.genetic import SymbolicRegressor
from sklearn.model_selection import train_test_split

# Daten generieren
m_values = np.linspace(0.001, 0.01, 25)   # 50 m-Werte
V_values = np.linspace(10, 100, 25)       # 50 V-Werte

data = []
for m in m_values:
    for V in V_values:
        c = m / V  # Konzentration in mol/mL (optional V/1000, wenn in L)
        if c <= 0:
            continue
        pOH = -np.log10(c)
        # pH = 14 - pOH
        data.append([m, V, pOH])

# In NumPy-Arrays konvertieren
data = np.array(data)
X = data[:, :2]  # m und V als Features
y = data[:, 2]   # pH als Target

# In Trainings- und Testdaten aufteilen
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)



# Geschützter log10
def _log10(x):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x) > 1e-10, np.log10(np.abs(x)), 0.0)

log10_func = make_function(function=_log10, name='log10', arity=1)

# Geschützter log2
def _log2(x):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x) > 1e-10, np.log2(np.abs(x)), 0.0)

log2_func = make_function(function=_log2, name='log2', arity=1)

# Symbolic Regressor initialisieren
model = SymbolicRegressor(population_size=1000, generations=20,
                          tournament_size=20, stopping_criteria=0.01,
                          const_range=(0, 10), init_depth=(2, 6),
                          init_method='half and half',
                          parsimony_coefficient=0.02,  # Komplexitätsstrafe
                          function_set = ( 'sqrt', 'log', 'abs', 'neg', 'inv', 'max', 'min', 'mul', 'div', log10_func, log2_func,  'add', 'sub'),
                          metric='mean absolute error', verbose=1)

# Modell trainieren
model.fit(X_train, y_train)

# Ergebnisse anzeigen
print(f"\nBest symbolic expression:\n{model._program}")
print(f"\nModel R² score: {model.score(X_test, y_test)}")



