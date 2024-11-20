import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso

df = pd.read_csv("measurements.csv", encoding="latin-1", sep=",")
print(df.head())
x = df["distance"].str.replace(',', '.').astype(float).values
y = df["consume"].str.replace(',', '.').astype(float).values
X = x.reshape(-1, 1)  

# divide los datos 80%/20% de forma aleatoria donde el 20% será para pruebas
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)

##### esto es importante para los modelos de regularización como Lasso y Ridge, ver en profundid este tema... 
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
X_train = scale.fit_transform(X_train)
X_test = scale.transform(X_test)

# de aplica Regresiones con penalización Lasso (L1) valor abs. --> 0 a los valores de entrenamiento 80%  train
lasso = Lasso()
lasso.fit(X_train, y_train)
# predice los valores de prueba
y_pred = lasso.predict(X_test)
# se calculan alugunas métricas errores de la predicción
error_abs_medio = mean_absolute_error(y_test, y_pred)
error_cuadratico_medio = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Error Absoluto Medio: ", error_abs_medio)
print("Error Cuadrático Medio: ", error_cuadratico_medio)
print("R2: ", r2)

# entrenar el parámetro de regularización alpha
param_grid = {'alpha': [0.0001, 0.001, 0.1, 1, 10, 100]}
lasso_cv = GridSearchCV(lasso, param_grid, cv=3, n_jobs=-1) # cv=3 es el cross-value para cuantas veces pasará por el paramtetro alpha
lasso_cv.fit(X_train, y_train)

y_pred2 = lasso_cv.predict(X_test)
# se calculan alugunas métricas errores de la predicción
error_abs_medio = mean_absolute_error(y_test, y_pred2)
error_cuadratico_medio = mean_squared_error(y_test, y_pred2)
r2 = r2_score(y_test, y_pred2)
print("Error Absoluto Medio: ", error_abs_medio)
print("Error Cuadrático Medio: ", error_cuadratico_medio)
print("R2: ", r2)
mejor_estimador = lasso_cv.best_estimator_
print("Mejor estimador: ", mejor_estimador)
# creo un nuevo lasso ocupando el mejor estimador alpha
lasso3 = Lasso(alpha=mejor_estimador.alpha)
lasso3.fit(X_train, y_train)
intercept = lasso3.intercept_
print("Intercepto: ", intercept)
coefs = lasso3.coef_
print("Coeficientes: ", coefs)




