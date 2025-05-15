import numpy as np
import matplotlib.pyplot as plt

from naturkonstanten import pi, epsilon_0

# 2 Punktladungen mit zufälligen Werten
N = 200
rng = np.random.default_rng(42)
q1 = rng.uniform(-2, 2, N)
q2 = rng.uniform(-2, 2, N)
r = rng.uniform(0.5, 5.0, N)   # Kein r=0, damit keine Singularität

# Feature-Array: Spalten sind [q1, q2, r]
X = np.column_stack([q1, q2, r])
y = ( q1 * q2) / (4 * pi * epsilon_0 * r)

print(X, y)
plt.plot(X, y,"x")
plt.show()

