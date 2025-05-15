# SOLL:
( x0 * x1 / x2 ) * 8 987 551 786.170797
(1/(4 * pi * epsilon_0))

# Daten ohne Rauschen:
Best symbolic expression: complexity                                            7
loss                                          161844.81
equation                  (x1 * (x0 / x2)) * 8.987 552e9
score                                         16.222036
sympy_format                      x1*x0*8987552000.0/x2
lambda_format    PySRFunction(X=>x1*x0*8987552000.0/x2)
Name: 3, dtype: object



# Verrauschte Daten:
Best symbolic expression: complexity                                            7
loss                               280569130000000000.0
equation                  ((x1 * x0) / x2) * 9.006 403e9
score                                          2.161409
sympy_format                      x1*x0*9006403000.0/x2
lambda_format    PySRFunction(X=>x1*x0*9006403000.0/x2)
Name: 4, dtype: object