import pandas as pd
import numpy as np
from sklearn.preprocessing import minmax_scale, scale
#Estes caras nao funcionaram para valores categoricos (strings), 
#na hora de fazer a imputacao
# from sklearn.experimental import enable_iterative_imputer
# from sklearn.impute import IterativeImputer

#Deixei o codigo do que nao funcionou por aqui, vai que preciso voltar atras
# df = pd.read_csv("HDI.csv")
# X=df.to_numpy()
# Take a look at the first few rows
# print (df.head())
#print (df['Working poor at PPP$3.10 a day (%) 2004-2013'])
# print (X)
# imp_mean = IterativeImputer(random_state=0)
# imp_mean.fit(X)
# imp_mean.transform(X)
# print(X)

#Entao encontrei essa funcao no git que faz a mediana e funciona
#com strings na imputacao.
#https://stackoverflow.com/questions/25239958/impute-categorical-missing-values-in-scikit-learn 
from sklearn.base import TransformerMixin
class DataFrameImputer(TransformerMixin):
    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value 
        in column.

        Columns of other types are imputed with mean of column.

        """
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)

#Lendo os dados de treinamento e aplicando imputacao nos dados faltantes
df = pd.read_csv("HDI.csv")
print(df)
names = df['Country']
print(names)
cols = [0,1,2,3]
df.drop(df.columns[cols],axis=1,inplace=True)
df=DataFrameImputer().fit_transform(df)
print(df.head)
print(type(df))
X=scale(df)
# X = (df - np.mean(df, axis=0)) / np.std(df, axis=0)
# X = df.values
print(type(X))
print(X)

#criando o modelo SOM a partir de https://github.com/JustGlowing/minisom
som_shape = (15, 15)
from minisom import MiniSom    
# som = MiniSom(size, size, len(df.columns), neighborhood_function='gaussian', sigma=1.5,random_seed=1)
# som.pca_weights_init(df.to_numpy())
# som.train_random(df.to_numpy(), 100000,verbose=True) # trains the SOM with 100 iterations
som = MiniSom(som_shape[0], som_shape[1], len(X[0]), neighborhood_function='gaussian', sigma=1.5,random_seed=1)
som.pca_weights_init(X)
som.train_random(X, 1000000,verbose=True) # trains the SOM with 100 iterations

# feature_names = ['','Id','HDI Rank','HDI','Life expectancy','Mean years of schooling','Gross national income (GNI) per capita','GNI per capita rank minus HDI rank','Change in HDI rank 2010-2015','Average annual HDI growth 1990-2000','Average annual HDI growth 2000-2010','Average annual HDI growth 2010-2015','Average annual HDI growth 1990-2015','Gender Development Index value','Gender Development Index Group','Human Development Index (HDI) Female','Human Development Index (HDI) Male','Life expectancy at birth Female','Life expectancy at birth Male','Mean years of schooling Female','Mean years of schooling Male','Estimated gross national income per capita Female','Estimated gross national income per capita Male','Share of seats in parliament (% held by women)','Population with at least some secondary education % (2005-2015) Female','Population with at least some secondary education % (2005-2015) Male','Labour force participation rate (% ages 15 and older) Female ','Total Population (millions) 2015','Total Population (millions) 2030','Population Average annual growth 2000/2005 (%) ','Population Average annual growth 2010/2015 (%) ','Population Urban 2015 %','Population Under age 5 (millions) 2015','Population Ages 15–64 (millions) 2015','Population Ages 65 and older (millions) 2015','Population Median age (years) 2015','Dependency Ration Young age (0–14) /(per 100 people ages 15–64)','Dependency Ratio Old age (65 and older) /(per 100 people ages 15–64)','Total fertility rate (birth per woman) 2000/2005','Total fertility rate (birth per woman) 2000/2007','Infants exclusively breastfed (% ages 0–5 months) 2010-2015','Infants lacking immunization DTP (% of one-year-olds)','Infants lacking immunization Measles (% of one-year-olds)','Child malnutrition Stunting (moderate or severe) 2010-2015','"Mortality rates Infant (per 1','000 live births) 2015"','"Mortality rates Under-five (per 1','000 live births) 2015"','"Mortality rates Female Adult (per 1','000 live births) 2014"','"Mortality rates Male Adult (per 1','000 live births) 2014"','"Deaths due to Malria (per 100','000 people) "','"Deaths due to Tuberculosis (per 100','000 people) "','"HIV prevalence',' adult (% ages 15–49)"','Life expectancy at age 59 (years) 2010/2015','"Physicians  (per 10','000 people) 2001-2014"','Public health expenditure (% of GDP) 2014','Employment to population ratio (% ages 15 and older) ','Labour force participation rate (% ages 15 and older)','Employment in agriculture (% of total employment) 2010-2014','Employment in services (% of total employment) 2010- 2014','Total Unemployment (% of labour force) 2015','Unemployment Youth (% ages 15-24) 2010-2014','Unemployment Youth not in school or employment (% ages 15-24) 2010-2014','Vulnerable employment (% of total employment) 2005-2014','Child labour  (% ages 5-14) 2009-2015','Working poor at PPP$3.10 a day (%) 2004-2013','Mandatory paid maternity leave (days)','Old-age pension recipients  (% of statutory pension age population) 2004-2013','Internet users','Internet users (% 2010 -2015)','Inequality-adjusted HDI (IHDI)','Inequality-adjusted HDI (IHDI) Over loss(%)','Difference from HDI rank','Coefficient of human inequality','Inequality in life expectancy (%) 2010-2015','Inequality-adjusted life expectancy index','Inequality in education(%)','Inequality-adjusted education index','Inequality in income (%)','Inequality-adjusted income index','Income inequality (Quintile ratio) 2010-2015','Income inequality (Palma ratio) 2010-2015','Income inequality (Gini coefficient) 2010-2015']


#saida grafica dos pesos SOM apos treinamento
import plotly.graph_objects as go
# win_map = som.win_map(df.to_numpy())
win_map = som.win_map(X)
# size2=som.distance_map().shape[0]
qualities=np.empty(som_shape)
qualities[:]=np.NaN
for position, values in win_map.items():
    qualities[position[0], position[1]] = np.mean(abs(values-som.get_weights()[position[0], position[1]]))
layout = go.Layout(title='quality plot')
fig = go.Figure(layout=layout)
fig.add_trace(go.Heatmap(z=qualities, colorscale='Viridis'))
fig.show()


#organizando dados do treinamento
data=X
print(type(data))
print(data.shape)
# each neuron represents a cluster
winner_coordinates = np.array([som.winner(x) for x in data]).T
print("winners",type(winner_coordinates))
print(winner_coordinates.shape)
print(winner_coordinates)
cluster_index = np.ravel_multi_index(winner_coordinates, som_shape)
print(type(cluster_index))
print(cluster_index.shape)
print(cluster_index)

#outra saida grafica dos pesos obtidos
import matplotlib.pyplot as plt
# plotting the clusters using the first 2 dimentions of the data
print("data",data)
for c in np.unique(cluster_index):
	# print (c,cluster_index==c, end=",")
	plt.scatter(data[cluster_index == c, 0],
				data[cluster_index == c, 1], label='cluster='+str(c), alpha=.7)
				# data[cluster_index == c, 1], label=names[c], alpha=.7)
# plotting centroids
# for centroid in som.get_weights():
#     plt.scatter(centroid[:, 0], centroid[:, 1], marker='x', 
#                 s=30, linewidths=.5, color='k', label='centroid')
plt.legend()
plt.show()


#realizando a clusterizacao dos pesos SOM com KMeans
from sklearn.cluster import KMeans
kmeans= KMeans (n_clusters=3)
kmeans.fit(data,sample_weight=cluster_index)
pred=kmeans.predict(data)
print(pred)
print(data[:,0].shape)
print('data[:,0]',data[:,0])
plt.scatter(data[:,0],data[:,1], s=100,c=kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], s=50, marker='x',c='red')
plt.show()