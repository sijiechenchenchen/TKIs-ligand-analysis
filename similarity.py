import sdf_tools as st
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns; sns.set_theme()
from matplotlib import pyplot as plt 



if __name__=='__main__': 

	### define sdf file path ###
	active_sdf ='sdfs/active_less_than10.sdf'
	inactive_sdf ='sdfs/inactive.sdf'
	### calcualte similarity ###
#	sim_matrix1=st.get_cosin_similarity(train_sdf,test_sdf,feature='Morgan')
	sim_matrix2=st.get_tanimoto_similarity(active_sdf ,inactive_sdf,feature='Morgan')
	sim_matrix3=st.get_tanimoto_similarity(active_sdf ,active_sdf ,feature='Morgan')
	sim_matrix4=st.get_tanimoto_similarity(inactive_sdf ,inactive_sdf ,feature='Morgan')
	### plot ###
	fig, axs = plt.subplots(1,3, sharex=False, sharey=False,figsize=(30,20))

	ax1 = sns.heatmap(sim_matrix2, ax=axs[0],cmap="YlGnBu",vmin=0, vmax=1)
	ax2 = sns.heatmap(sim_matrix3, ax=axs[1],cmap="YlGnBu",vmin=0, vmax=1)
	ax3 = sns.heatmap(sim_matrix4, ax=axs[2],cmap="YlGnBu",vmin=0, vmax=1)
	plt.savefig("tanimoto_similarity.png",dpi=300,transparent=True)	
#	plt.show()