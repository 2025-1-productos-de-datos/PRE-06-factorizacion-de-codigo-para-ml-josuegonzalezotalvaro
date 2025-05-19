import pandas as pd

# descarga de datos
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df = pd.read_csv(url, sep=";")

# guardar el dataframe en un archivo CSV en data
df.to_csv("data/winequality-red.csv", index=False)
