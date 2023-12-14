import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm


# 1. Naprawa danych (zamiana na wartości liczbowe, itp.)

# Metody selekcji zmiennych
# 1. Metoda eliminacji
# 2. Metoda wstecznej eliminacji (backward elimination) (zachlanna minimalizacja)
# 3. PCA

# Analiza zbudowanego modelu
# 1. Test F - test wariancji
# 2. Regresja Ridge
# 3. Pokazanie korelacji
# 4. R-squared

# imports = pd.read_csv("./datasets/imports-85.csv")
# imports.info()
#
# label_encoder = LabelEncoder()
# for column in imports.columns:
#     if imports[column].dtype == 'object':
#         imports[column] = label_encoder.fit_transform(imports[column])
#
# imports.info()
#
# imports_dr = imports.dropna()
# imports_dr.describe()
#
# imports_dr = imports_dr.drop('engine-location', axis=1)
#
# plt.figure(figsize=(12,10))
# corr_matrix = imports_dr.corr()
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.1f', linewidths=.5)
# plt.title("Wykres korelacji")
# plt.show()
#
#
#
# m1 = sm.OLS(imports_dr['price'], sm.add_constant(imports_dr.drop('price', axis=1))).fit()
# print(m1.summary())
#
# m2 = sm.OLS(imports_dr['price'], sm.add_constant(imports_dr.drop(['price','length'], axis=1))).fit()
# print(m2.summary())
# # m2 = sm.OLS(imports_dr['price'], sm.add_constant(imports_dr[['symboling', 'normalized-losses','make','fuel-type','aspiration','num-of-doors','body-style', 'drive-wheels','wheel-base','width','height','curb-weight','engine-type','num-of-cylinders','engine-size','fuel-system','bore','stroke','compression-ratio','horsepower','peak-rpm','city-mpg','highway-mpg' ]])).fit()
# # print(m2.summary())
#
# m3 = sm.OLS(imports_dr['price'], sm.add_constant(imports_dr.drop(['price', 'length', 'symboling'], axis=1))).fit()
# print(m3.summary())
#
# m4 = sm.OLS(imports_dr['price'], sm.add_constant(imports_dr.drop(['price', 'length', 'symboling', 'bore'], axis=1))).fit()
# print(m4.summary())
#
# m5 = sm.OLS(imports_dr['price'], sm.add_constant(imports_dr.drop(['price', 'length', 'symboling', 'bore', 'highway-mpg'], axis=1))).fit()
# print(m5.summary())
#
# m6 = sm.OLS(imports_dr['price'], sm.add_constant(imports_dr.drop(['price', 'length', 'symboling', 'bore', 'highway-mpg', 'peak-rpm'], axis=1))).fit()
# print(m6.summary())
#
# m7 = sm.OLS(imports_dr['price'], sm.add_constant(imports_dr.drop(['price', 'length', 'symboling', 'bore', 'highway-mpg', 'peak-rpm', 'wheel-base'], axis=1))).fit()
# print(m7.summary())
#
# m8 = sm.OLS(imports_dr['price'], sm.add_constant(imports_dr.drop(['price', 'length', 'symboling', 'bore', 'highway-mpg', 'peak-rpm', 'wheel-base', 'city-mpg'], axis=1))).fit()
# print(m8.summary())
#
# m9 = sm.OLS(imports_dr['price'], sm.add_constant(imports_dr.drop(['price', 'length', 'symboling', 'bore', 'highway-mpg', 'peak-rpm', 'wheel-base', 'city-mpg', 'normalized-losses'], axis=1))).fit()
# print(m9.summary())
#
# m10 = sm.OLS(imports_dr['price'], sm.add_constant(imports_dr.drop(['price', 'length', 'symboling', 'bore', 'highway-mpg', 'peak-rpm', 'wheel-base', 'city-mpg', 'normalized-losses', 'horsepower'], axis=1))).fit()
# print(m10.summary())
#
# m11 = sm.OLS(imports_dr['price'], sm.add_constant(imports_dr.drop(['price', 'length', 'symboling', 'bore', 'highway-mpg', 'peak-rpm', 'wheel-base', 'city-mpg', 'normalized-losses', 'horsepower', 'num-of-doors'], axis=1))).fit()
# print(m11.summary())
#
# m12 = sm.OLS(imports_dr['price'], sm.add_constant(imports_dr.drop(['price', 'length', 'symboling', 'bore', 'highway-mpg', 'peak-rpm', 'wheel-base', 'city-mpg', 'normalized-losses', 'horsepower', 'num-of-doors', 'engine-type'], axis=1))).fit()
# print(m12.summary())
#
# m13 = sm.OLS(imports_dr['price'], sm.add_constant(imports_dr.drop(['price', 'length', 'symboling', 'bore', 'highway-mpg', 'peak-rpm', 'wheel-base', 'city-mpg', 'normalized-losses', 'horsepower', 'num-of-doors', 'engine-type', 'engine-size'], axis=1))).fit()
# print(m13.summary())
#
# m14 = sm.OLS(imports_dr['price'], sm.add_constant(imports_dr.drop(['price', 'length', 'symboling', 'bore', 'highway-mpg', 'peak-rpm', 'wheel-base', 'city-mpg', 'normalized-losses', 'horsepower', 'num-of-doors', 'engine-type', 'engine-size', 'height'], axis=1))).fit()
# print(m14.summary())
#
# m15 = sm.OLS(imports_dr['price'], sm.add_constant(imports_dr.drop(['price', 'length', 'symboling', 'bore', 'highway-mpg', 'peak-rpm', 'wheel-base', 'city-mpg', 'normalized-losses', 'horsepower', 'num-of-doors', 'engine-type', 'engine-size', 'height', 'body-style'], axis=1))).fit()
# print(m15.summary())
#
# m16 = sm.OLS(imports_dr['price'], sm.add_constant(imports_dr.drop(['price', 'length', 'symboling', 'bore', 'highway-mpg', 'peak-rpm', 'wheel-base', 'city-mpg', 'normalized-losses', 'horsepower', 'num-of-doors', 'engine-type', 'engine-size', 'height', 'body-style', 'fuel-system'], axis=1))).fit()
# print(m16.summary())
# # m16 = sm.OLS(imports_dr['price'], sm.add_constant(imports_dr[['normalized-losses','make','fuel-type','aspiration', 'drive-wheels','width','curb-weight','num-of-cylinders','stroke','compression-ratio']])).fit()
# # print(m16.summary())
#
# print("R-squared dla modelu m1: ", m1.rsquared)
# print("R-squared adjusted dla modelu m1: ", m1.rsquared_adj)
# print("R-squared dla modelu m16: ", m16.rsquared)
# print("R-squared adjusted dla modelu m16: ", m16.rsquared_adj)
#
# anova = anova_lm(m1,m2,m3,m4,m5,m6,m7,m8,m16)
# print(anova)


wines = pd.read_csv('./datasets/wine.csv')
wines.info()
print(wines.describe())

plt.figure(figsize=(12,10))
corr_matrix = wines.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.1f', linewidths=.5)
plt.title("Wykres korelacji")
plt.show()

m1 = sm.OLS(wines['Wine'], sm.add_constant(wines.drop('Wine', axis=1))).fit()
print(m1.summary())
m2 = sm.OLS(wines['Wine'], sm.add_constant(wines.drop(['Wine', 'Mg'], axis=1))).fit()
print(m2.summary())
m3 = sm.OLS(wines['Wine'], sm.add_constant(wines.drop(['Wine', 'Mg', 'Proanth'], axis=1))).fit()
print(m3.summary())
m4 = sm.OLS(wines['Wine'], sm.add_constant(wines.drop(['Wine', 'Mg', 'Proanth', 'Hue'], axis=1))).fit()
print(m4.summary())

f,p_value,_ = m1.compare_f_test(m4)
print(p_value)

