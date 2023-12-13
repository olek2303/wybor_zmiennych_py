import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

iris = pd.read_csv("./datasets/Iris.csv")
print(iris.head())
print(iris.shape)

# 1. Naprawa danych (zamiana na wartości liczbowe, itp.)

# Metody selekcji zmiennych
# 1. Metoda eliminacji
# 2. Metoda wstecznej eliminacji (backward elimination) (zachlanna minimalizacja)
# 3. PCA

# Analiza zbudowanego modelu
# 1. Test Anova - test wariancji
# 2. Regresja Ridge
# 3. Pokazanie korelacji
# 4. R-squared
