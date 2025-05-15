from gplearn.genetic import SymbolicRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Beispiel-Daten generieren (z.B. y = x^2 + x + 1)
X, y = make_regression(n_samples=100, n_features=1, noise=0.1)
X = X.flatten()  # Flatten für 1D-Daten
y = X**2 + X + 1

# Daten in Trainings- und Testdaten aufteilen
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Symbolic Regressor initialisieren
model = SymbolicRegressor(population_size=1000, generations=20, tournament_size=20,
                          stopping_criteria=0.01, const_range=(0, 10), init_depth=(2, 6),
                          init_method='half and half', function_set=('add', 'sub', 'mul', 'div'),
                          metric='mean absolute error')

# Modell trainieren
model.fit(X_train.reshape(-1, 1), y_train)

# Vorhersagen und das beste gefundene Modell ausgeben
print(f"Best symbolic expression: {model._program}")
y_pred = model.predict(X_test.reshape(-1, 1))
print(f"Model score: {model.score(X_test.reshape(-1, 1), y_test)}")
