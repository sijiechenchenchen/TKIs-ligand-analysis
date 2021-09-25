import sdf_tools as st
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns; sns.set_theme()
from matplotlib import pyplot as plt 
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def evaluate_components(fp_list):
    res = []
    for n_comp in tqdm(range(2,50)):
        pca = PCA(n_components=n_comp)
        crds = pca.fit_transform(fp_list)
        var = np.sum(pca.explained_variance_ratio_)
        res.append([n_comp,var])
    return res

if __name__=='__main__': 

	### define sdf file path ###
	active_sdf ='sdfs/active_less_than10.sdf'
	inactive_sdf ='sdfs/inactive.sdf'
	allo='ASD_druggable'
	allo_all='ASD_Release_201909_3D'
	drug_all='sdfs/all_drugs.sdf'

	pca = PCA(n_components=20)
	data1 = st.read_sdf_mol(active_sdf)
	data2 = st.read_sdf_mol(inactive_sdf) 
	data3 = st.read_mol2_mol(allo)
	data4 = st.read_mol2_mol(allo_all)
	data5 = st.read_sdf_mol(drug_all)


	fps1=st.molsfeaturizer(data1)
	pca1=pca.fit(fps1)
	pca1=pca.transform(fps1)
	#fps2=st.molsfeaturizer(data2)
	#pca2=pca.fit_transform(fps2)
#	fps3=st.molsfeaturizer(data3)
#	pca3=pca.fit_transform(fps3)
	fps4=st.molsfeaturizer(data4)
	pca4=pca.transform(fps4)
	fps5=st.molsfeaturizer(data5)
	pca5=pca.transform(fps5)

	pca_ensemble=np.vstack((pca1,pca4,pca5))
	tsne_ensemble = TSNE(n_components=2, perplexity=30.0, n_iter=1000, verbose=1).fit_transform(pca_ensemble)
#	tsne1 = TSNE(n_components=2, perplexity=30.0, n_iter=1000, verbose=1).fit(pca1)
	tsne1 = TSNE(n_components=2, perplexity=30.0, n_iter=1000, verbose=1).fit_transform(pca1)
	#tsne2 = TSNE(n_components=2).fit_transform(pca2)
	#tsne3 = TSNE(n_components=2).fit_transform(pca3)
	tsne4 = TSNE(n_components=2, perplexity=30.0, n_iter=1000, verbose=1).fit_transform(pca4)
	tsne5 = TSNE(n_components=2, perplexity=30.0, n_iter=1000, verbose=1).fit_transform(pca5)

#	plt.scatter(tsne4[:,0],tsne4[:,1],color='purple',alpha=0.3,linewidths=0.5)
#	plt.scatter(tsne5[:,0],tsne5[:,1],color='grey',alpha=0.3,linewidths=0.5)
#	plt.scatter(tsne1[:,0],tsne1[:,1],color='red')
	#plt.scatter(tsne2[:,0],tsne2[:,1],color='black')
	#plt.scatter(tsne3[:,0],tsne3[:,1],color='pink')
	plt.scatter(tsne_ensemble[32:-3967][:,0],tsne_ensemble[32:-3967][:,1],color='purple',alpha=0.3,linewidths=0.5)
	plt.scatter(tsne_ensemble[-3968:][:,0],tsne_ensemble[-3968:][:,1],color='grey',alpha=0.3,linewidths=0.5)
	plt.scatter(tsne_ensemble[0:31][:,0],tsne_ensemble[0:31][:,1],color='red')
	

	### plot ###
	plt.savefig("tsne-all.png",dpi=300,transparent=True)	
#	plt.show()