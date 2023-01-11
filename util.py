from collections import defaultdict
import networkx as nx
import numpy as np
import os
import time
import random
import sys

class tools:
	def calc_centrality(self,G):
		node_centrality_values = defaultdict(list)
		if 	os.path.exists(G.name+'/degree.txt') and\
			os.path.exists(G.name+'/pagerank.txt') and\
			os.path.exists(G.name+'/betweenness.txt') and\
			os.path.exists(G.name+'/eigenvector.txt') and\
			os.path.exists(G.name+'/closeness.txt'):
			print "Loading centrality rankings from file..."
			for key in ["degree","pagerank","betweenness","eigenvector","closeness"]:
				with open(G.name+"/"+key+".txt") as r:
					content = r.read().splitlines()
				r.close()
				node_centrality_values[key] = np.asarray(content).astype(float)
		else:
			if not os.path.exists(G.name+'/degree.txt'):
				print "Calculating degree..."
				start_time = time.time()
				node_centrality_values["degree"] = nx.degree_centrality(G)
				print "degree\t\tTime: ", time.time()-start_time
				fw =  open(G.name+'/degree.txt',"w")
				for node in node_centrality_values["degree"]:
					fw.write("%s\n" % node_centrality_values["degree"][node])
				fw.close()

			if not os.path.exists(G.name+'/pagerank.txt'):
				print "Calculating pagerank..."
				start_time = time.time()
				node_centrality_values["pagerank"] = nx.pagerank(G)
				print "pagerank\t\tTime: ", time.time()-start_time
				fw =  open(G.name+'/pagerank.txt',"w")
				for node in node_centrality_values["pagerank"]:
					fw.write("%s\n" % node_centrality_values["pagerank"][node])
				fw.close()

			if not os.path.exists(G.name+'/betweenness.txt'):
				print "Calculating betweenness..."
				start_time = time.time()
				node_centrality_values["betweenness"] = nx.betweenness_centrality(G)
				print "betweenness\t\tTime: ", time.time()-start_time
				fw =  open(G.name+'/betweenness.txt',"w")
				for node in node_centrality_values["betweenness"]:
					fw.write("%s\n" % node_centrality_values["betweenness"][node])
				fw.close()

			if not os.path.exists(G.name+'/eigenvector.txt'):
				print "Calculating eigenvector..."
				start_time = time.time()
				node_centrality_values["eigenvector"] = nx.eigenvector_centrality_numpy(G)
				print "eigenvector\t\tTime: ", time.time()-start_time
				fw =  open(G.name+'/eigenvector.txt',"w")
				for node in node_centrality_values["eigenvector"]:
					fw.write("%s\n" % node_centrality_values["eigenvector"][node])
				fw.close()

			if not os.path.exists(G.name+'/closeness.txt'):
				print "Calculating closeness..."
				start_time = time.time()
				node_centrality_values["closeness"] = nx.closeness_centrality(G)
				print "closeness\t\tTime: ", time.time()-start_time
				fw =  open(G.name+'/closeness.txt',"w")
				for node in node_centrality_values["closeness"]:
					fw.write("%s\n" % node_centrality_values["closeness"][node])
				fw.close()

		return node_centrality_values

	"""
	Sort the centrality rankings for nodes
	"""	
	def adj_list_sorted_centrality(self, node_centrality_values, adj_lists):
		sorted_adj_lists = defaultdict(set)
		centrality_sorted_adj_dic = defaultdict()  
		for centrality in node_centrality_values:
			for node in adj_lists:
				sorted_adj_lists[node] = set(sorted(adj_lists[node], key=lambda x: node_centrality_values[centrality][x]))
			centrality_sorted_adj_dic[centrality] = sorted_adj_lists
		centrality_sorted_adj_dic["uniform"] = adj_lists
		return centrality_sorted_adj_dic

	"""
	Generate centrality biased random walks for unsupervised training.
	"""	
	def centrality_biased_random_walk(self,G,node_centrality_values):
		if 	os.path.exists(G.name+'/degree_walk.txt') 	and\
			os.path.exists(G.name+'/pagerank_walk.txt') and\
			os.path.exists(G.name+'/betweenness_walk.txt') and\
			os.path.exists(G.name+'/eigenvector_walk.txt') and\
			os.path.exists(G.name+'/closeness_walk.txt'):
			print "Centrality Biased Random Walks exists, continue..."
		else:
			print "Generating Centrality Biased Random Wlaks..."
			# centrality biased random walks
			adj_prob = defaultdict()
			num_walks = 5
			WALK_LEN = 50
			for sampler in node_centrality_values:
				adj_prob[sampler] = defaultdict(list)
				if sampler == "uniform":
					break
				# init neighbours sampling prob accroding to centrality value
				for node in G.nodes():
					neighbors = [i for i in G.neighbors(node)]
					adj_prob[sampler][node] = node_centrality_values[sampler][neighbors]
				# random walk
				out_file = G.name+'/'+sampler+'_walk.txt'
				pairs = []
				for count, node in enumerate(G.nodes()):
					if G.degree(node) == 0:
						continue
					for i in range(num_walks):# for each node, generating #num of walks 
						curr_node = node
						for _ in range(WALK_LEN):
							neighbors = [i for i in G.neighbors(curr_node)]
							adj_prob_cur_node = [float(i) for i in adj_prob[sampler][curr_node]]
							adj_prob_cur_node_norm = [i/sum(adj_prob_cur_node) for i in adj_prob_cur_node]
							next_node = np.random.choice(neighbors,p = adj_prob_cur_node_norm)
							if curr_node != node:
								pairs.append((node,curr_node))
							curr_node = next_node
				print("Done",sampler," walks for", count, "nodes, writing to file ...")
				with open(out_file, "w") as fp:
					fp.write("\n".join([str(p[0]) + "\t" + str(p[1]) for p in pairs]))

	"""
	negative sampling
	To-do:
		* use sub_network node sets, e.g., u2u_nodes, u2b_nodes
		* use nodes sets for each set in a each sub_graph. e.g., uSet in u2u, uSet in u2b and bSet in u2b. 
		* above 2 will solve sample the most fair negative samples
		* (Later) add addtional u2u links related to uSet of the u2b network. (curenttly we only use u2u links between users in the uSet of u2b) 
	"""
	def negtive_sampling(self, G, net_type, uSet, vSet, k, node_centrality_values, file_path): 
		nodes_prob = defaultdict()
		neg_neighbors = []

		print "Start negative sampling..."
		#out_file = file_path + G.name + net_type + '_neg_samples.txt'
		out_file = file_path + net_type + '_neg_samples.txt'

		# out_negative1k = file_path + G.name + 'u2b_1k_samples.csv'
		out_negative1k = file_path + 'u2b_1k_samples.csv'

		# if file exists, just load
		if not os.path.exists(out_file) or not os.path.exists(out_negative1k):

			with open(out_file, "w") as fp:
				start = time.clock()
				nodes_prob = defaultdict(list)
				for node in G.nodes():
					if net_type == 'u2u':
						candi_set = set(nx.non_neighbors(G,node)).intersection(uSet)
						nonNeighbors = list(set([x for x in candi_set]))

					if net_type == 'u2b':
						if node in uSet:
							candi_set = set(nx.non_neighbors(G,node)).intersection(vSet)
						else:
							candi_set = set(nx.non_neighbors(G,node)).intersection(uSet)
						nonNeighbors = list(set([x for x in candi_set]))

					nodes_prob[node] = np.random.choice(nonNeighbors, size=k)

					fp.write("\t".join([str(x) for x in nodes_prob[node]]))
					fp.write("\n")
				print "negative sampling for "+G.name+" used: ",time.clock()-start
			fp.close()

			if not os.path.exists(out_negative1k) and net_type == 'u2b':
				with open(out_negative1k, "w") as wp:
					start = time.clock()
					nodes_prob = defaultdict(list)
					for node in uSet:
						nonNeighbors = list(set([x for x in nx.non_neighbors(G,node)]).intersection(vSet)) #hit ratio test
						nodes_prob[node] = np.random.choice(nonNeighbors, size=1000)
						wp.write(str(node)+",")
						for sample in nodes_prob[node]:
							wp.write(str(sample)+",")
						wp.write("\n")
					print "negative 1k sampling for "+G.name+" used: ",time.clock()-start				

			with open(out_file) as r:
				content = r.read().splitlines()
				for line in content:
					info = line.split("\t")
					neg_neighbors.append([int(x) for x in info])
			r.close()
			
		else:
			with open(out_file) as r:
				content = r.read().splitlines()
				for line in content:
					info = line.split("\t")
					neg_neighbors.append([int(x) for x in info])
			print "Loaded fromn existing files"
		
		neg_neighbors = np.array(neg_neighbors)
		return neg_neighbors

	"""
	auc_test data generate 
	"""
	def gen_auc_test_data(self,G,N):

		pos =  np.array([list(e) for e in G.edges()])[0:N]
		# print pos
		# pos = [np.array(x) for x in G.edges()][0:N]
		# print pos
		ones = np.ones(N)
		# random.shuffle(pos)
		pos = np.column_stack((pos,ones)).astype(int)

		u = pos[:,0]
		v = []
		for _ in u:
			v.append(np.random.choice([ x for x in nx.non_neighbors(G,_)]))
		v = np.asarray(v)
		zeros = np.zeros(N)
		negs = np.column_stack((u,v,zeros)).astype(int)
		auc_test = np.concatenate((pos,negs))

		return auc_test

