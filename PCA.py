from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler



df = pd.read_csv("/home/rohan/Documents/calciumwave/calciumwave/noPLCd_noSOC_ampl_speeds",delimiter=' ')

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

keep = ['Vratio','Buffer', 'ip3r', 'ip3_rest', '#diam'
'wave']

x = df.loc[:,keep].values # Separating out the target
y = df.loc[:,['wave']].values # Standardizing the features
#x = StandardScaler().fit_transform(x)


pca = PCA()
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component '+str(i) for i in range(0,len(keep))])

finalDf = pd.concat([principalDf, df[['wave']]], axis = 1)

#principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])




fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_zlabel('Principal Component 3', fontsize = 15)
# ax.scatter(principalDf['principal component 1']
#            , principalDf['principal component 2']
#            , principalDf['principal component 3']
#            , s = 50)
targets = [1, 0]
colors = ['r', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['wave'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , finalDf.loc[indicesToKeep, 'principal component 3']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()



