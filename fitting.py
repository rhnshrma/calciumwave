from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler



df = pd.read_csv("/home/rohan/Documents/calciumwave/noPLCd_noSOC_ampl_speeds",delimiter=' ')

def invert(x):
	return(1/x)

def prod(x,y):
	return(x*y)

def ratio(x,y):
	return(x/y)

postive_features = ['Vratio','ip3r','ip3_rest']
product_features = []

for i,f1 in enumerate(postive_features[:-1]):
	for f2 in postive_features[i+1:]:
		product_features.append([f1,f2])
	df[f1+'/Buffer'] = ratio(df[f1],df['Buffer'])

for i,feat in enumerate(product_features):
	df[feat[0]+'*'+feat[1]] = prod(df[feat[0]],df[feat[1]])
	df[feat[0]+'*'+feat[1]+'/Buffer'] = ratio(df[feat[0]+'*'+feat[1]],df['Buffer'])

df['Buffer-1'] = invert(df['Buffer'])
df['kdeg-1'] = invert(df['kdeg'])  
#df = df.loc[(df['speed'] > 0)]

#Filter to speed>0
#df = df.loc[(df['speed'] > 0)]

df = df.assign(wave=np.zeros(1000))

for i in df.index :
	if df['speed'][i]>0 : df['wave'][i]=1
	else : df['wave'][i]=0

# features = ['Vratio','Buffer','ip3r','ip3_rest','wave'] 
df = df[np.isfinite(df).all(1)]

features = ['Vratio','Buffer', 'ip3r', 'ip3_rest', 'Vratio/Buffer', 'ip3r/Buffer',
       'ip3_rest/Buffer', 'Vratio*ip3r', 'Vratio*ip3r/Buffer',
       'Vratio*ip3_rest', 'Vratio*ip3_rest/Buffer', 'ip3r*ip3_rest',
       'ip3r*ip3_rest/Buffer', 'Buffer-1','kdeg-1', 'wave']

keep = ['Vratio/Buffer', 'Vratio*ip3_rest/Buffer','ip3r','kdeg-1','ip3_rest','ip3r/Buffer','ip3r*ip3_rest/Buffer',
'wave']

x = df.loc[:,keep].values # Separating out the target
y = df.loc[:,['wave']].values # Standardizing the features
#x = StandardScaler().fit_transform(x)
Y = y.ravel()
X = x
clf = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3)) 
clf.fit(X, Y) 