import random

from sklearn.metrics import silhouette_samples, silhouette_score
import k_means
import mixture_models
from matplotlib import pyplot as plt
from sklearn.preprocessing import scale
from statsmodels.graphics.gofplots import qqplot
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.cm as cm
import numpy as np
import pyclustertend

# Cargando datos
dataset = pd.read_csv('minute_weather.csv')

# setting seed
np.random.seed(0)
random.seed(0)
RANDOM_STATE = 0

ANALISIS = False

# Analizando datos
if ANALISIS:
    print(dataset.head())
    print(dataset.describe(include='all'))

    # Limpiando datos
    for col in dataset.columns:
        if col == 'hpwren_timestamp':
            dataset[col] = pd.to_datetime(dataset[col], format='%Y-%m-%d %H:%M:%S')
            dataset[col].hist(bins=100)
            plt.title('Histogram of {}'.format(col))
            plt.show()
            
        else:
            # grafica QQ
            qqplot(dataset[col], line='s')
            plt.title(col)
            plt.show()
            # find outliers
            dataset[col].plot(kind='box')
            plt.show()

# removing outliers
dataset['rain_accumulation'] = dataset['rain_accumulation'].apply(lambda x: x if x < 2 else np.nan)
dataset['rain_duration'] = dataset['rain_duration'].apply(lambda x: x if x < 500 else np.nan)

# normalize data
dataset['rain_duration'] = (dataset['rain_duration'] - dataset['rain_duration'].mean()) / dataset['rain_duration'].std()

# Viendo variables sin outliers
if ANALISIS:
    for col in ['rain_accumulation', 'rain_duration']:
        qqplot(dataset[col], line='s')
        plt.title(col)
        plt.show()
        # find outliers
        dataset[col].plot(kind='box')
        plt.show()

# Selecting features
dataset.dropna(inplace=True)
dataset_cuantitative = dataset[['air_pressure', 'air_temp', 'avg_wind_direction',
                            'avg_wind_speed', 'max_wind_direction', 'max_wind_speed',
                            'min_wind_direction', 'min_wind_speed', 'rain_accumulation', 
                            'rain_duration', 'relative_humidity']]

# Calculando numero adecuado de clusters
if ANALISIS:
    # Metodo de codo
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, max_iter=300)
        kmeans.fit(dataset_cuantitative)
        wcss.append(kmeans.inertia_)
    plt.plot(range(1, 11), wcss)
    plt.title('Grafico de codo')
    plt.xlabel('No. Clusters')
    plt.ylabel('Puntaje')
    plt.show()

    # seleccionar un porcentaje de datos
    percentage = 0.25
    dataset_norm_sample = dataset_cuantitative.sample(frac=percentage, random_state=RANDOM_STATE)

    # Metodo hopkins
    print(pyclustertend.hopkins(dataset_norm_sample, len(dataset_norm_sample)))

    # seleccionar un porcentaje de datos
    percentage = 0.005
    dataset_norm_sample = dataset_cuantitative.sample(frac=percentage, random_state=RANDOM_STATE)

    # Grafico VAT e iVAT
    pyclustertend.vat(dataset_norm_sample)
    plt.show()
    pyclustertend.ivat(dataset_norm_sample)
    plt.show()

N_CLUSTERS = 5

# Clustering
# K means
k_means.execute_kmeans(dataset_cuantitative, N_CLUSTERS, 300)

# Mixture models
mixture_models.execute_mixture_models(dataset_cuantitative, N_CLUSTERS)
