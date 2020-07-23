from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler



df = pd.read_csv("/home/rohan/Documents/calciumwave/calciumwave/noPLCd_noSOC_ampl_speeds",delimiter=' ')
df = df[np.isfinite(df).all(1)]

#df = df.loc[(df['speed'] > 0)]

#Filter to speed>0
#df = df.loc[(df['speed'] > 0)]

df = df.assign(wave=np.zeros(999))

for i in df.index :
	if df['speed'][i]>0 : df['wave'][i]=1
	else : df['wave'][i]=0

features = ['#diam','Vratio','Buffer','ip3r','ip3_rest','wave'] 

x = df.loc[:, features].values # Separating out the target
y = df.loc[:,['wave']].values # Standardizing the features
x = StandardScaler().fit_transform(x)


pca = PCA()
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2', 'principal component 3', 'principal component 4', 'principal component 5', 'principal component 6'])

finalDf = pd.concat([principalDf, df[['wave']]], axis = 1)

#principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])




fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(111, projection='3d')
# ax.set_xlabel('Principal Component 1', fontsize = 15)
# ax.set_ylabel('Principal Component 2', fontsize = 15)
# ax.set_zlabel('Principal Component 3', fontsize = 15)
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


