import matplotlib.pyplot as plt 
from matplotlib import style
style.use('ggplot')
import numpy as np 
from sklearn.cluster import KMeans
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


# klasifikujemo ljude u prezivele i poginule samo na osnovu ostalih kolona i 
# poredimo sa kolonom za prezivele i gledamo koliko smo uboli

df = pd.read_excel('titanic.xls')
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

clf = KMeans(n_clusters=2)
clf.fit(X)

correct = 0
for i in range(len(X)):
	predict_me = np.array(X[i].astype(float))
	predict_me = predict_me.reshape(-1,len(predict_me))
	prediction = clf.predict(predict_me)
	if prediction[0] == y[i]:
		correct += 1

print(correct/len(X))



## primercic
#X = np.array([[1,2],
#		   	 [1.5,1.8],
#			 [5,8],
#			 [8,8],
#			 [1,0.6],
#			 [9,11]])

## plt.scatter(X[:,0], X[:,1], s=50, linewidth=5)
## plt.show()

#clf = KMeans(n_clusters=2)
#clf.fit(X)

#centroids = clf.cluster_centers_
#labels = clf.labels_

#colors = 10*["g.","r.","c.","b."]

#for i in range(len(X)):
#	plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize=20)
#plt.scatter(centroids[:,0],centroids[:,1],marker='x', s=150,linewidth=5)
#plt.show()