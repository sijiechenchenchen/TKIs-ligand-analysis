import sdf_tools as st
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns; sns.set_theme()
from matplotlib import pyplot as plt 
import numpy as np
from sklearn.decomposition import PCA
if __name__=='__main__': 

	### define sdf file path ###
	active_sdf ='sdfs/active_less_than10.sdf'
	inactive_sdf ='sdfs/inactive.sdf'
	allo='ASD_druggable'
	allo_all='ASD_Release_201909_3D'

	pca = PCA(n_components=2)
	data1 = st.read_sdf_mol(active_sdf)
	data2 = st.read_sdf_mol(inactive_sdf) 
	data3 = st.read_mol2_mol(allo)
	data4 = st.read_mol2_mol(allo_all)

	fps1=st.molsfeaturizer(data1)
	pca1=pca.fit_transform(fps1)
	fps2=st.molsfeaturizer(data2)
	pca2=pca.fit_transform(fps2)
	fps3=st.molsfeaturizer(data3)
	pca3=pca.fit_transform(fps3)
	fps4=st.molsfeaturizer(data4)
	pca4=pca.fit_transform(fps4)

	plt.scatter(pca4[:,0],pca4[:,1],color='grey',alpha=0.3)
	plt.scatter(pca1[:,0],pca1[:,1],color='red')
	plt.scatter(pca2[:,0],pca2[:,1],color='green')
	plt.scatter(pca3[:,0],pca3[:,1],color='blue')
	plt.scatter(pca4[:,0],pca3[:,1],color='grey',alpha=0.5)
	
	### plot ###
	plt.savefig("pca-all.png",dpi=300,transparent=True)	
#	plt.show()