## For Python 3
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
# import graph_tool
import imp
# de_groot_models = imp.load_source("de_groot_models","2019_10_26_de_groot_models.py")
de_groot_models = __import__('2019_10_26_de_groot_models')

def get_retweet_NW(
	msg_list
	):
	"""
	Created:4-1-2016
	Modified:04-03-2020
	Builds the retweet network of a set of tweets contained in files. Returns
	a graph networkx file with the screen names as labels instead of a dictionary.
	Modified to make it much simpler. Can be used with mongo data.
	"""
	print ("Starting get_retweet_NW function")
	G = nx.DiGraph()
	for k,msg in enumerate(msg_list):
		if k % 10000 == 0:
			print (k)
		if 'retweeted_status' in msg:
			msg_date = msg["created_at_dt"]
			o_id = msg['user']['id']
			o_name = msg['user']['screen_name']
			retweeted_status = msg['retweeted_status']
			d_id = retweeted_status['user']['id']
			d_name = retweeted_status['user']['screen_name']
			## Create nodes and edges
			G = add_node_to_twitter_nw(G,o_id,o_name,msg_date,'out')
			G = add_node_to_twitter_nw(G,d_id,d_name,msg_date,'in')
			G = add_edge_to_nw(G,o_id,d_id,w=1)
	return G

def add_node_to_twitter_nw(G,node_id,node_name,msg_date,in_out):
	"""
	Created: 22-9-2016
	Modified: 22-9-2016
	Adds a node to a network of twitter users in a rational way: If the node
	exists, it checks if it has changed its screen name and in that case 
	stores it in a string. It also adds some more relevant attributes.
	"""
	msg_date_str = msg_date.strftime("%Y-%m-%d %H:%M:%S")
	if not G.has_node(node_id):
		G.add_node(node_id,label=node_name,old_labels="",times=msg_date_str,in_out=in_out)
	else:
		G.nodes[node_id]['times'] = G.nodes[node_id]['times'] + "," + msg_date_str
		G.nodes[node_id]['in_out'] = G.nodes[node_id]['in_out'] + "," + in_out
		if not node_name == G.nodes[node_id]['label']:
			G.nodes[node_id]['old_labels'] = G.nodes[node_id]['old_labels'] + "," + G.nodes[node_id]['label']
			G.nodes[node_id]['label'] = node_name
	return G

def add_edge_to_nw(G,o,d,w=1):
    """
    Created:10-11-2015
    Modified:10-11-2015
    Helper function to add edges in a coherent way.
    """
    if G.has_edge(o,d):
        G[o][d]['weight'] += w
    else:
        G.add_edge(o,d,weight=w)
    return G

def better_to_undirected(G):
	"""
	Created:12-11-2015
	Modified:12-11-2015
	The to_undirected() function of networkx returns a copy of the original 
	directed graph where the weights of the edges are more or less arbitrary:
	Either the new weight of an edge (a,b) is the weight of the old edge (a,b)
	or of the old edge (b,a). 
	In this function the final weight is the sum of both.
	"""
	G_UD = nx.Graph()
	G_UD.add_nodes_from(G,data=True)
	for (o,d) in G.edges():
		# if G[o][d].has_key('weight'):
		w = G[o][d]['weight']
		# else:
			# raise Exception("You may want to use nx.to_undirected better, since this network has no weights")
		G_UD = add_edge_to_nw(G_UD,o,d,w)
	return G_UD

##############################################################################
## graph-tool utilitiez
##############################################################################

def convert_networkx_to_graphtool(G_nx):
	"""
	Created: 2020-03-13
	Modified: 2020-03-13
	Converts a networkx directed graph to a graph-tool undirected graph.
	"""
	G_nx_UD = misc_tools.better_to_undirected(G_nx)
	G_nx_UD_GCC = polarization_visualization_tools.get_GCC(G_nx_UD)
	idx_to_node = list(G_nx_UD_GCC.nodes)
	node_to_idx = {n:i for i,n in enumerate(idx_to_node)}
	g_gt = graph_tool.Graph(directed=False)
	g_gt.add_vertex(len(idx_to_node))
	weight_prop = g_gt.new_edge_property("int")
	for o, d in G_nx_UD_GCC.edges:
	    w = G_nx_UD_GCC[o][d]["weight"]
	    v1 = g_gt.vertex(node_to_idx[o])
	    v2 = g_gt.vertex(node_to_idx[d])
	    e = g_gt.add_edge(v1,v2)
	    weight_prop[e] = w
	g_gt.edge_properties["weight"] = weight_prop
	return idx_to_node, node_to_idx, g_gt

def include_HBM_communities_in_graph(
	g_gt,
	state_gt,
	prefix = "community_level"):
	state_gt.print_summary()
	arry_vertices = g_gt.get_vertices()
	levels = state_gt.get_levels()
	print("Numero de niveles: ", len(levels))

	for i in range(len(levels)):
	    string = prefix + str(i)
	    vprop = g_gt.new_vertex_property("double")
	    g_gt.vertex_properties[string]= vprop 

	for i,num in enumerate(arry_vertices):
	    v = g_gt.vertex(num)
	    r = levels[0].get_blocks()[v]
	    string = prefix + "0"
	    g_gt.vertex_properties[string][v] = r
	    for i in range(1,len(levels)):
	        string = prefix + str(i)
	        r = levels[i].get_blocks()[r]
	        g_gt.vertex_properties[string][v] = r

	print("ChecK", g_gt.list_properties())
	return g_gt

def include_parties_in_graph(
	g_gt, 
	idx_to_node, 
	node_to_idx,
	party_nodes
	# cs_acc_id,
	# psoe_acc_id,
	# pod_acc_id,
	# pp_acc_id
	):
	parties_vals = ["-" for _ in idx_to_node]
	err_cntr = 0
	for party, node_lst in party_nodes.items():
		err_cntr = 0
		for node in node_lst:
			try:
				idx = node_to_idx[node]
			except KeyError:
				err_cntr += 1
				continue
			parties_vals[idx] = party
		print (party, err_cntr, len(node_lst))
	# for node in cs_acc_id:
	#     try:
	#         idx = node_to_idx[node]
	#     except KeyError:
	#         err_cntr += 1
	#         continue
	#     parties_vals[idx] = "Cs"
	# print (err_cntr, len(cs_acc_id))
	# err_cntr = 0
	# for node in pp_acc_id:
	#     try:
	#         idx = node_to_idx[node]
	#     except KeyError:
	#         err_cntr += 1
	#         continue
	#     parties_vals[idx] = "PP"
	# print (err_cntr, len(pp_acc_id))
	# err_cntr = 0
	# for node in psoe_acc_id:
	#     try:
	#         idx = node_to_idx[node]
	#     except KeyError:
	#         err_cntr += 1
	#         continue
	#     parties_vals[idx] = "PSOE"
	# print (err_cntr, len(psoe_acc_id))
	# err_cntr = 0
	# for node in pod_acc_id:
	#     try:
	#         idx = node_to_idx[node]
	#     except KeyError:
	#         err_cntr += 1
	#         continue
	#     parties_vals[idx] = "Podemos"
	# print (err_cntr, len(pod_acc_id))
	parties_prop = g_gt.new_vertex_property("string",vals=parties_vals)
	g_gt.vertex_properties["party"] = parties_prop
	return g_gt

def include_screen_names_and_twitID_in_graph(
	g_gt, 
	idx_to_node, 
	node_to_idx, 
	act_dct,
	rt_dct
	):
	screen_names = []
	for node in idx_to_node:
	    try:
	        sn = act_dct[node]["screen_name"]
	    except KeyError:
	        try:
	            sn = rt_dct[node]["screen_name"]
	        except KeyError:
	            print ("Screen name not found")
	            sn = "?"
	    screen_names.append(sn)
	screen_name_prop = g_gt.new_vp("string",vals=screen_names)
	uid_prop = g_gt.new_vp("long long",vals=node_to_idx)
	g_gt.vertex_properties["screen_name"] = screen_name_prop
	g_gt.vertex_properties["twitter_id"] = uid_prop
	return g_gt

##############################################################################
## Elite selection tools
##############################################################################

def rt_participation_filtering(
	rt_min,
	part_min,
	idx_to_node, 
	rt_dct, 
	participation_dct):
	rt_es = []
	prt_es = []
	selected_nodes_gt = []
	for i, usr in enumerate(idx_to_node):
	    try:
	        rti = rt_dct[usr]["rt"]
	    except KeyError:
	        continue
	    try:
	        parti = participation_dct[usr]["partic_aggr"]
	    except KeyError:
	        continue
	    rt_es.append(rti)
	    prt_es.append(parti)
	    if rti > rt_min and parti > part_min:
	        selected_nodes_gt.append(i)
	rt_es = np.array(rt_es)
	prt_es = np.array(prt_es)
	msk = np.logical_and(rt_es>rt_min,prt_es>part_min)
	print ("Total selected nodes:", sum(msk), len(selected_nodes_gt))

	## Figure
	plt.semilogy(prt_es,rt_es,"o",ms=1)
	plt.semilogy(prt_es[msk],rt_es[msk],"Xr",ms=5)
	plt.xlabel("Participation/days")
	plt.ylabel("Retweets")
	return selected_nodes_gt	

def include_elite_prop_and_get_elite_graph(g_gt, idx_to_node, selected_nodes_gt):
	## Build elite subgraph
	elite_vals = np.array([False for _ in idx_to_node])
	elite_vals[selected_nodes_gt] = True
	elite_prop = g_gt.new_vertex_property("bool",vals=elite_vals)
	g_gt.vertex_properties["elite"] = elite_prop
	g_gt_elite = graph_tool.GraphView(g_gt, vfilt=elite_prop).copy()
	g_gt_elite.purge_vertices()
	return g_gt, g_gt_elite

def get_poles_nodes(
	g_gt_elite,
	level,
	pole_comm_dct,
	prefix="community_level"):
	poles_nodes_dct = {k:[] for k in pole_comm_dct.keys()}
	for v in g_gt_elite.vertices():
		for pole, comm_lst in pole_comm_dct.items():
			for comm in comm_lst:
			    if g_gt_elite.vp[f"{prefix}{level}"][v] == comm:
			        poles_nodes_dct[pole].append(g_gt_elite.vp.twitter_id[v])
	return poles_nodes_dct

def assign_poles_to_nodes(
	poles_corresp,
	poles_nodes_dct):
	diff_poles_corresp = set(poles_corresp.keys())
	diff_poles_in_nodes_dct = set(poles_nodes_dct.keys())
	if diff_poles_corresp != diff_poles_in_nodes_dct:
		print (f"Warning!! Different poles in poles-index dict and in\
poles-nodes dict:\npoles_corresp{diff_poles_corresp}\
\npoles_nodes_dct{diff_poles_in_nodes_dct}")
		if diff_poles_corresp-diff_poles_in_nodes_dct!=set():
			raise Exception("There are poles in poles_corresp that are not\
in poles_nodes_dct. This would cause an excessive number of \
dimensions in the polarization computation.")

	poles_coords = de_groot_models.get_simplex_vertex(len(poles_corresp)-1)
	elite_nodes = []
	elite_dct = {}
	for pole, nodes in poles_nodes_dct.items():
	    print (pole, len(nodes))
	    idx = poles_corresp[pole]
	    coords = poles_coords[idx]
	    elite_nodes.extend(nodes)
	    for n in nodes:
	        elite_dct[n] = coords
	print (len(elite_dct))
	return elite_dct, elite_nodes

def myvar_v3(data):
    data = np.array(data)
    m = np.mean(data,axis=0)
    comps = data-m
    comps = comps*comps
    assert comps.shape == data.shape
    return np.sum(comps)/(1.0*len(data))

def filter_usrs_by_activity(
	data,
	act_dct,
	usrs_order,
	act_min):
	filtered = []
	for i, n in enumerate(usrs_order):
		if act_dct[n]["act"] >= act_min:
			filtered.append(data[i])
	return filtered