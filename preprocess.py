from collections import defaultdict
import itertools
import os
import pickle
import random

import networkx as nx
from networkx.algorithms import bipartite as bi
import numpy as np
from scipy import sparse
from scipy.io import loadmat
from sklearn.preprocessing import normalize

import graph


def save_homogenous_graph_to_file(A, datafile, index_row, index_item):
    (M, N) = A.shape
    csr_dict = A.__dict__
    data = csr_dict.get("data")
    indptr = csr_dict.get("indptr")
    indices = csr_dict.get("indices")
    col_index = 0

    with open(datafile, 'w') as fw:
        for row in range(M):
            for col in range(indptr[row], indptr[row + 1]):
                r = row
                c = indices[col]
                fw.write(str(index_row.get(r)) + "\t" + str(index_item.get(c)) + "\t" + str(data[col_index]) + "\n")
                col_index += 1


def calculate_centrality(G, uSet, bSet, mode='hits'):
    authority_u = {}
    authority_v = {}

    if mode == 'degree_centrality':
        a = nx.degree_centrality(G)
    else:
        h, a = nx.hits(G)

    max_a_u, min_a_u, max_a_v, min_a_v = 0, 100000, 0, 100000

    for node in G.nodes():
        if node in uSet:
            if max_a_u < a[node]:
                max_a_u = a[node]
            if min_a_u > a[node]:
                min_a_u = a[node]
        if node in bSet:
            if max_a_v < a[node]:
                max_a_v = a[node]
            if min_a_v > a[node]:
                min_a_v = a[node]

    for node in G.nodes():
        if node in uSet:
            if max_a_u - min_a_u != 0:
                authority_u[node] = (float(a[node]) - min_a_u) / (max_a_u - min_a_u)
            else:
                authority_u[node] = 0
        if node in bSet:
            if max_a_v - min_a_v != 0:
                authority_v[node] = (float(a[node]) - min_a_v) / (max_a_v - min_a_v)
            else:
                authority_v[node] = 0

    return authority_u, authority_v


def get_random_walks_restart(datafile): # hits_dict, percentage, maxT, minT
    G = graph.load_edgelist(datafile, undirected=True)
    print("Folded HIN ==> number of nodes: {}".format(len(G.nodes())))
    print("walking...")
    # walks = graph.build_deepwalk_corpus_random(G, hits_dict, percentage=percentage, maxT = maxT, minT = minT, alpha=0)
    walks = graph.build_deepwalk_corpus(G, None, 5, rand=random.Random())
    print("walking...ok")

    return G, walks

def innovation_get_random_walks_restart(datafile, prob):
    G = graph.load_edgelist(datafile, undirected=True)
    print("Folded HIN ==> number of nodes: {}".format(len(G.nodes())))
    print("walking...")
    walks = graph.innovation_build_deepwalk_corpus(G, None, 5, prob)
    print("walking...ok")

    return G, walks


def generate_bipartite_folded_walks(path, history_u_lists, history_v_lists, edge_list_uv, edge_list_vu):
    BiG = nx.Graph()
    node_u = history_u_lists.keys()
    node_v = history_v_lists.keys()
    sorted(node_u)
    sorted(node_v)

    BiG.add_nodes_from(node_u, bipartite=0)
    BiG.add_nodes_from(node_v, bipartite=1)
    BiG.add_weighted_edges_from(edge_list_uv + edge_list_vu)
    A = bi.biadjacency_matrix(BiG, node_u, node_v, dtype=np.float64, weight='weight', format='csr')

    row_index = dict(zip(node_u, itertools.count()))
    col_index = dict(zip(node_v, itertools.count()))

    index_row = dict(zip(row_index.values(), row_index.keys()))
    index_item = dict(zip(col_index.values(), col_index.keys()))

    AT = A.transpose()
    fw_u = os.path.join(path, "homogeneous_u.dat")
    fw_v = os.path.join(path, "homogeneous_v.dat")
    save_homogenous_graph_to_file(A.dot(AT), fw_u, index_row, index_row)
    save_homogenous_graph_to_file(AT.dot(A), fw_v, index_item, index_item)

    # TODO
    authority_u, authority_v = calculate_centrality(BiG, node_u, node_v)

    # TODO
    G_u, walks_u = get_random_walks_restart(fw_u)
    G_v, walks_v = get_random_walks_restart(fw_v)

    return G_u, walks_u, G_v, walks_v


def innovation(path, history_u_lists, history_v_lists, edge_list_uv, edge_list_vu):
    BiG = nx.Graph()
    node_u = sorted(history_u_lists.keys())
    node_v = sorted(history_v_lists.keys())
    n_users = len(node_u)
    n_items = len(node_v)

    BiG.add_nodes_from(node_u, bipartite=0)
    BiG.add_nodes_from(node_v, bipartite=1)
    BiG.add_weighted_edges_from(edge_list_uv + edge_list_vu)
    A = bi.biadjacency_matrix(BiG, node_u, node_v, dtype=np.uint32, weight='weight', format='csr')
    AT = A.transpose()

    first_order_user_A = A.dot(AT)
    second_order_user_A = first_order_user_A.dot(first_order_user_A)

    first_order_user_A.setdiag(0)
    second_order_user_A.setdiag(0)

    first_order_item_A = AT.dot(A)
    second_order_item_A = first_order_item_A.dot(first_order_item_A)

    first_order_item_A.setdiag(0)
    second_order_item_A.setdiag(0)

    folded_user_A = sparse.csr_array((n_users, n_users), dtype=np.float32)
    folded_user_A[second_order_user_A > 0] = 0.25
    folded_user_A[first_order_user_A > 0] = 0.5
    normalize(folded_user_A, norm='l1', axis=1, copy=False)

    folded_item_A = sparse.csr_array((n_items, n_items), dtype=np.float32)
    folded_item_A[second_order_item_A > 0] = 0.25
    folded_item_A[first_order_item_A > 0] = 0.5
    normalize(folded_item_A, norm='l1', axis=1, copy=False)

    row_index = dict(zip(node_u, itertools.count()))
    col_index = dict(zip(node_v, itertools.count()))

    index_row = dict(zip(row_index.values(), row_index.keys()))
    index_item = dict(zip(col_index.values(), col_index.keys()))

    fw_u = os.path.join(path, "homogeneous_u.dat")
    fw_v = os.path.join(path, "homogeneous_v.dat")
    save_homogenous_graph_to_file(sparse.csr_matrix(
        folded_user_A), fw_u, index_row, index_row)
    save_homogenous_graph_to_file(sparse.csr_matrix(
        folded_item_A), fw_v, index_item, index_item)

    G_u, walks_u = innovation_get_random_walks_restart(fw_u, folded_user_A.toarray())
    G_v, walks_v = innovation_get_random_walks_restart(fw_v, folded_item_A.toarray())

    return G_u, walks_u, G_v, walks_v


def innovation2(path, history_u_lists, history_v_lists, edge_list_uv, edge_list_vu):
    BiG = nx.Graph()
    node_u = sorted(history_u_lists.keys())
    node_v = sorted(history_v_lists.keys())
    n_users = len(node_u)
    n_items = len(node_v)

    BiG.add_nodes_from(node_u, bipartite=0)
    BiG.add_nodes_from(node_v, bipartite=1)
    BiG.add_weighted_edges_from(edge_list_uv + edge_list_vu)
    A = bi.biadjacency_matrix(BiG, node_u, node_v, dtype=np.float64, weight='weight', format='csr')
    AT = A.transpose()

    first_order_user_A = A.dot(AT)
    second_order_user_A = first_order_user_A.dot(first_order_user_A)

    first_order_user_A = first_order_user_A.toarray()
    np.fill_diagonal(first_order_user_A, 0)

    second_order_user_A = second_order_user_A.toarray()
    np.fill_diagonal(second_order_user_A, 0)

    first_order_item_A = AT.dot(A)
    second_order_item_A = first_order_item_A.dot(first_order_item_A)

    first_order_item_A = first_order_item_A.toarray()
    np.fill_diagonal(first_order_item_A, 0)

    second_order_item_A = second_order_item_A.toarray()
    np.fill_diagonal(second_order_item_A, 0)

    def normalize(probs):
        prob_factor = 1 / sum(probs)
        return [prob_factor * p for p in probs]

    folded_user_A = np.ndarray(shape=(n_users, n_users), dtype=float)

    for i in range(n_users):
        folded_user_A[i] = normalize(second_order_user_A[i])

    folded_item_A = np.ndarray(shape=(n_items, n_items), dtype=float)

    for i in range(n_items):
        folded_item_A[i] = normalize(second_order_item_A[i])

    row_index = dict(zip(node_u, itertools.count()))
    col_index = dict(zip(node_v, itertools.count()))

    index_row = dict(zip(row_index.values(), row_index.keys()))
    index_item = dict(zip(col_index.values(), col_index.keys()))

    fw_u = os.path.join(path, "homogeneous_u.dat")
    fw_v = os.path.join(path, "homogeneous_v.dat")
    save_homogenous_graph_to_file(sparse.csr_matrix(
        folded_user_A), fw_u, index_row, index_row)
    save_homogenous_graph_to_file(sparse.csr_matrix(
        folded_item_A), fw_v, index_item, index_item)

    G_u, walks_u = innovation_get_random_walks_restart(fw_u, folded_user_A)
    G_v, walks_v = innovation_get_random_walks_restart(fw_v, folded_item_A)

    return G_u, walks_u, G_v, walks_v


def preprocess(path):
    uSet_u2u = set()
    uSet_u2b = set()
    bSet_u2b = set()

    social_adj_lists = defaultdict(set)
    history_u_lists = defaultdict(list)
    history_v_lists = defaultdict(list)

    history_ur_lists = defaultdict(list)
    history_vr_lists = defaultdict(list)

    G = nx.Graph()
    G.name = 'epinions'

    ratings_f = loadmat(path + 'epinions/rating.mat')['rating']
    trust_f = loadmat(path + 'epinions/trustnetwork.mat')['trustnetwork']

    users = []

    for s in ratings_f:
        uid = s[0]
        users.append(uid)

    offset = len(set(users))

    for s in ratings_f:
        uid = s[0]
        iid = s[1] + offset
        rating = s[3]
        uSet_u2b.add(uid)
        bSet_u2b.add(iid)
        G.add_edge(uid, iid, type='u2b', rating=rating)

    for s in trust_f:
        uid = s[0]
        fid = s[1]
        uSet_u2u.add(uid)
        uSet_u2u.add(fid)
        G.add_edge(uid, fid, type='u2u')

    print(G)
    print("uSet of u2u, size: " + str(len(uSet_u2u)))
    print("uSet of u2b, size: " + str(len(uSet_u2b)))
    print("bSet of u2b, size: " + str(len(bSet_u2b)))

    # Relabeling nodes to consecutive integers
    G = nx.convert_node_labels_to_integers(G, first_label=0, ordering='default', label_attribute="name")

    node_names = nx.get_node_attributes(G, 'name')
    inv_map = {v: k for k, v in node_names.items()}

    # Converting nodes in the sets to the relabeled nodes
    uSet_u2u = set([inv_map.get(name) for name in uSet_u2u])
    uSet_u2b = set([inv_map.get(name) for name in uSet_u2b])
    bSet_u2b = set([inv_map.get(name) for name in bSet_u2b])

    edge_list_uv = []
    edge_list_vu = []

    for node in G:
        for nbr in G[node]:
            if G[node][nbr]['type'] == 'u2u':
                social_adj_lists[node].add(nbr)

            if G[node][nbr]['type'] == 'u2b':
                r = G[node][nbr]['rating'] - 1

                if node in uSet_u2b and nbr in bSet_u2b:
                    history_u_lists[node].append(nbr)
                    history_v_lists[nbr].append(node)
                    history_ur_lists[node].append(r)
                    history_vr_lists[nbr].append(r)
                    edge_list_uv.append((node, nbr, r))
                    edge_list_vu.append((nbr, node, r))

                if nbr in uSet_u2b and node in bSet_u2b:
                    history_u_lists[nbr].append(node)
                    history_v_lists[node].append(nbr)
                    history_ur_lists[nbr].append(r)
                    history_vr_lists[node].append(r)
                    edge_list_uv.append((nbr, node, r))
                    edge_list_vu.append((node, nbr, r))

    G_u, walks_u, G_v, walks_v = innovation(path, history_u_lists, history_v_lists, edge_list_uv, edge_list_vu)
    # G_u, walks_u, G_v, walks_v = generate_bipartite_folded_walks(path, history_u_lists, history_v_lists, edge_list_uv,
    #                                                              edge_list_vu)

    # uSet_u2u = list(uSet_u2u)
    # random.shuffle(uSet_u2u)
    # size = len(uSet_u2u)
    # randomized_users = set(uSet_u2u[:int(0.8 * size)])

    # train_data = []
    # test_data = []

    # for (u, v) in G.edges():
    #     if G[u][v]['type'] == 'u2b':
    #         r = G[u][v]['rating'] - 1

    #         if u in randomized_users:
    #             if u in uSet_u2b:
    #                 train_data.append((u, v, r))
    #             else:
    #                 train_data.append((v, u, r))
    #         else:
    #             if u in uSet_u2b:
    #                 test_data.append((u, v, r))
    #             else:
    #                 test_data.append((v, u, r))

    data = []
    for (u, v) in G.edges():
        if G[u][v]['type'] == 'u2b':
            r = G[u][v]['rating'] - 1

            if u in uSet_u2b:
                data.append((u, v, r))
            else:
                data.append((v, u, r))

    random.shuffle(data)

    size = len(data)
    train_data = data[:int(0.8 * size)]
    test_data = data[int(0.8 * size):]

    # TODO: Validation set

    with open(path + 'dataset.pkl', 'wb') as f:
        pickle.dump(train_data, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(test_data, f, pickle.HIGHEST_PROTOCOL)

    train_u, train_v, train_r, test_u, test_v, test_r = [], [], [], [], [], []

    for u, v, r in train_data:
        train_u.append(u)
        train_v.append(v)
        train_r.append(r)

    for u, v, r in test_data:
        test_u.append(u)
        test_v.append(v)
        test_r.append(r)

    ratings_list = [0, 1, 2, 3, 4]

    _social_adj_lists = defaultdict(set)
    _history_u_lists = defaultdict(list)
    _history_v_lists = defaultdict(list)

    _history_ur_lists = defaultdict(list)
    _history_vr_lists = defaultdict(list)
    _train_u, _train_v, _train_r, _test_u, _test_v, _test_r = [], [], [], [], [], []

    social_id_dic = {v: k for k, v in dict(enumerate(social_adj_lists.keys())).items()}
    user_id_dic = {v: k for k, v in dict(enumerate(history_u_lists.keys())).items()}
    item_id_dic = {v: k for k, v in dict(enumerate(history_v_lists.keys())).items()}

    for u in history_u_lists:
        _history_u_lists[user_id_dic[u]] = [item_id_dic[v] for v in history_u_lists[u]]

    for v in history_v_lists:
        _history_v_lists[item_id_dic[v]] = [user_id_dic[u] for u in history_v_lists[v]]

    for u in history_ur_lists:
        _history_ur_lists[user_id_dic[u]] = history_ur_lists[u]

    for v in history_vr_lists:
        _history_vr_lists[item_id_dic[v]] = history_vr_lists[v]

    for u in social_adj_lists:
        _social_adj_lists[social_id_dic[u]] = [social_id_dic[us] for us in social_adj_lists[u]]

    for u, v, r in train_data:
        if u in user_id_dic.keys() and v in item_id_dic.keys():
            _train_u.append(user_id_dic[u])
            _train_v.append(item_id_dic[v])
            _train_r.append(r)

    for u, v, r in test_data:
        if u in user_id_dic.keys() and v in item_id_dic.keys():
            _test_u.append(user_id_dic[u])
            _test_v.append(item_id_dic[v])
            _test_r.append(r)

    _walks_u = defaultdict(list)
    _walks_v = defaultdict(list)

    for u in walks_u:
        _walks_u[user_id_dic[u]] = [user_id_dic[us] for us in walks_u[u]]
    for v in walks_v:
        _walks_v[item_id_dic[v]] = [item_id_dic[vs] for vs in walks_v[v]]

    with open(path + 'list.pkl', 'wb') as f:
        pickle.dump(_history_u_lists, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(_history_ur_lists, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(_history_v_lists, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(_history_vr_lists, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(_walks_u, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(_walks_v, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(_train_u, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(_train_v, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(_train_r, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(_test_u, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(_test_v, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(_test_r, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(_social_adj_lists, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(ratings_list, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    path = 'data/'
    preprocess(path)
