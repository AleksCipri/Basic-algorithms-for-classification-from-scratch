import matplotlib.pyplot as plt 
from matplotlib import style
style.use('ggplot')
import numpy as np 
from sklearn.cluster import MeanShift
from sklearn import preprocessing#, cross_validation
import pandas as pd 


'''
Pclass Pasanger Class (1,2,3)
Survival (0=No, 1=Yes)
Name
Sex
Age
Number of siblings/spouses aboard
number of parents/children aboard
ticket number
passanger fare (British pounds)
cabin
post of embarkation (C=Cherbourg, Q=Queenstown, S=Southampton)
lifeboat
body identification number
home/destinaton
'''

df = pd.read_excel('titanic.xls')
original_df = pd.DataFrame.copy(df)

df.drop(['body','name'], 1, inplace=True)
df.convert_objects(convert_numeric=True)
df.fillna(0, inplace=True)
print(df.head())

# konvertovanje podataka koji nisu brojevi u brojeve
def handle_non_numerical_data(df):
	columns = df.columns.values

	for column in columns:
		text_digit_vals = {}
		def convert_to_int(val):
			return text_digit_vals[val]
		if df[column].dtype != np.int64 and df[column].dtype !=np.float64:
			column_contents = df[column].values.tolist()
			unique_elements = set(column_contents)
			x = 0 
			for unique in unique_elements:
				if unique not in text_digit_vals:
					text_digit_vals[unique] = x
					x += 1
			df[column] = list(map(convert_to_int, df[column]))
	return df

df = handle_non_numerical_data(df)
# mozemo da se igramo sta izbacujemo da vidimo da li cmeo dobiti bolju tacnost
df.drop(['boat', 'sex'], 1, inplace=True)

X = np.array(df.drop(['survived'],1).astype(float))
# kada se doda scale popravi se tacnost ali se rotira koja grupa je 0 a koj a1 pa je nekad
# tacnost 30% a nekad 70% ali to znaci da smo tacni 70% puta svakako
# bez skaliranja tacnost nam je bila oko 45%
X = preprocessing.scale(X)
y = np.array(df['survived'])

clf = MeanShift()
clf.fit(X)

labels = clf.labels_
cluster_centers = clf.cluster_centers_

original_df['cluster_group'] = np.nan

for i in range(len(X)):
	original_df['cluster_group'].iloc[i] = labels[i]

n_clusters_ = len(np.unique(labels))

survival_rates = {}
for i in range(n_clusters_):
	temp_df = original_df[ (original_df['cluster_group']==float(i)) ]
	survival_cluster = temp_df[ (temp_df['survived']==1) ]
	survival_rate = len(survival_cluster)/len(temp_df)
	survival_rates[i] = survival_rate

print(survival_rates)
print(original_df[ (original_df['cluster_group']==0) ])
print(original_df[ (original_df['cluster_group']==0) ].describe())

cluster_0 = original_df[ (original_df['cluster_group']==0) ]
cluster_0_fc = cluster_0[ (cluster_0['pclass']==1) ]
print(cluster_0_fc.describe())






















