import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import pickle
# import graph_tool
# from graph_tool import topology as gt_topology
# from graph_tool import inference
import copy
import numpy as np
import imp
de_groot_models = imp.load_source("de_groot_models","2019_10_26_de_groot_models.py")
misc_tools = imp.load_source("misc_tools","2020_03_04_misc_tools.py")
polarization_visualization_tools = imp.load_source("polarization_visualization_tools","2019_11_23_polarization_visualization_tools.py")

def full_polarization_computations_v2(
	data_path_load,
	nw_load,
	poles_nodes_load,
	act_dct_load,
	data_path_save,
	elite_con_save,
	elite_dct_save,
	degroot_save,
	fig_path_save,
	fig_2d_save,
	fig_AvB_save,
	fig_data_dir,
	poles_corresp,
	degroot_tol = 1e-6,
	act_min = 30,
	media_ids_lst = []
	):
	## Load elite
	with open(data_path_load+poles_nodes_load,"rb") as f:
		poles_nodes_dct = pickle.load(f)
	## Check that there is no user overlap between parties 
	all_usrs = set()
	sum_len = 0
	for party, usrs in poles_nodes_dct.items():
		all_usrs.update(usrs)
		sum_len += len(usrs)
	assert sum_len == len(all_usrs)

	## Delete media user accounts
	for usr in media_ids_lst:
		for party in poles_nodes_dct:
			try:
				poles_nodes_dct[party].remove(usr)
				print ("Media user removed: ", party, usr)
			except:
				pass

	## Load network
	G = nx.read_weighted_edgelist(data_path_load+nw_load,delimiter=",",create_using=nx.DiGraph())
	## Compute Elite-connected network
	elite_dct, elite_nodes = misc_tools.assign_poles_to_nodes(
		poles_corresp,
		poles_nodes_dct)
	G_econ = de_groot_models.get_elite_connected(G, elite_nodes, max_path_len=10)
	print ("Elite-con network:", G_econ.order(), G.order(), nx.is_directed(G_econ))
	## Verify that the network is connected (althought not strongly connected)
	G_nx_UD = misc_tools.better_to_undirected(G_econ)
	G_nx_UD_GCC = polarization_visualization_tools.get_GCC(G_nx_UD)
	assert G_nx_UD_GCC.order() == G_econ.order() 

	## Save Elite-connected network
	nx.write_weighted_edgelist(G_econ,data_path_save+elite_con_save,delimiter=",")
	with open(data_path_save+elite_dct_save,"wb") as f:
		pickle.dump(elite_dct, f)

	## Compute degroot
	nodes_order= list(G_econ.nodes)
	order_dict= dict([ (v,i) for i,v in enumerate(nodes_order) ])
	degroot = de_groot_models.solve_model_Ax_iterative(
					G_econ,
					elite_dct,
					nodes_order = nodes_order,
					order_dict = order_dict,
					tol = degroot_tol
				)

	## Save degroot
	with open(data_path_save+degroot_save,"wb") as f:
		pickle.dump((nodes_order,order_dict,degroot),f)

	## Filter by activity
	with open(data_path_load+act_dct_load,"rb") as f:
		act_dct = pickle.load(f)
	degroot_act_filter = []
	kerr = 0
	for i, n in enumerate(nodes_order):
		try:
			if act_dct[n]["act"] >= act_min:
				degroot_act_filter.append(degroot[i])
		except KeyError:
			kerr += 1
	print ("kerr",kerr)
	print (len(degroot_act_filter), len(nodes_order))
	degroot_act_filter = np.array(degroot_act_filter)
	N_usrs = len(degroot_act_filter)
	print ("Final number of users: ", N_usrs)

	## Figures: orthog vs from pole and close to poles vs all
	fig_name_dct = {True:"Close",False:"All","orthogonal":"Ort","from_pole":"FrP"}
	proj_type = "orthogonal"
	project_close_only = True
	## Generate sufix name
	suf1 = fig_name_dct[proj_type]
	suf2 = fig_name_dct[project_close_only]
	suf = "_" + suf1 + "_" + suf2
	## Figure 2D projections
	fig = polarization_visualization_tools.vis_2d_projections(
	degroot_act_filter,
	show_2d_contours=True,
	show_center=True,
	show_barycenter=True,
	show_global_pca=True,
	project_close_only=project_close_only,
	save_fig_data=fig_path_save+fig_data_dir+fig_2d_save+suf,
	proj_type = proj_type,
	)
	fig.savefig(fig_path_save+fig_2d_save+suf+".pdf")
	fig.savefig(fig_path_save+fig_2d_save+suf+".png",dpi=600,transparent=True)

if __name__ == "__main__":
	## 2021-12-10, all paper computations

	## Load media accounts
	with open("../data/media_twitter_users/media_accs_cd.csv","r") as f:
		media_ids_lst = []
		for l in f.readlines():
			media_ids_lst.append(l[:-1])

	#### General elections 2015
	full_polarization_computations_v2(

		data_path_load="../partial_results/general_elections_2015_2016/",
		nw_load="rt_network_2015_cd.csv",
		poles_nodes_load="poles_2015_cd.p",
		act_dct_load="act_dict_2015_cd.p",

		data_path_save="../partial_results/general_elections_2015_2016/NEW/2015/",
		elite_con_save="rt_network_econ_cd.csv",
		elite_dct_save="elite_dct_cd.p",
		degroot_save="degroot_cd.p",

		fig_path_save="../figures/general_elections_2015_2016/NEW/2015/",
		fig_2d_save="degroot_2d_projs_2015",
		fig_AvB_save="degroot_AvB_2015",
		fig_data_dir="fig_data/",
		degroot_tol = 1e-1, ## For actual computation, we used 1e-6, but 1e-1 is enough
		act_min = 10,
		media_ids_lst = media_ids_lst,
		poles_corresp = {"PP":0,
			"PSOE":1,
			"Podemos":2,
			"Cs":3})

	### General elections 2019 28A
	full_polarization_computations_v2(

		data_path_load="../partial_results/general_elections_2019/",
		nw_load="rt_network_28A_cd.csv",
		poles_nodes_load="poles_28A_cd.p",
		act_dct_load="act_dict_28A_cd.p",

		data_path_save="../partial_results/general_elections_2019/NEW/28A/",
		elite_con_save="rt_network_econ_cd.csv",
		elite_dct_save="elite_dct_cd.p",
		degroot_save="degroot_cd.p",

		fig_path_save="../figures/general_elections_2019/NEW/28A/",
		fig_2d_save="degroot_2d_projs_28A",
		fig_AvB_save="degroot_AvB_28A",
		fig_data_dir="fig_data/",
		degroot_tol = 1e-1, ## For actual computation, we used 1e-6, but 1e-1 is enough
		act_min=30,
		media_ids_lst = media_ids_lst,
		poles_corresp = {"PP":0,
			"PSOE":1,
			"Podemos":2,
			"Cs":3,
			"Vox":4})