"""
Adithya Bhaskar, 2023.
This file contains utility functions that perform random number generation,
graph construction and other helper utilities.
"""

from config import *
from utils.globals import *

import os
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def get_node_indices_with_property(num_nodes, percent, avoid_first=False):
    """
    Returns a randomly chosen list of indices from [0, 1, ..., num_nodes-1]
    of size (percent/100.0)*num_nodes.
    if @avoid_first is set, then we ensure that 0 is not chosen (used
    to ensure that the attacker is not slow)
    """
    assert percent >= 0 and percent <= 100, "Percent should be between 0 and 100!"
    num_nodes_with_property = int(percent * num_nodes / 100)
    assert (not avoid_first) or num_nodes_with_property < num_nodes, \
        "Percent too high to have a fast attacker!"
    if avoid_first:
        num_nodes -= 1
    if num_nodes_with_property == 0:
        return []
    node_indices = np.random.permutation(num_nodes)[:num_nodes_with_property]
    node_indices = node_indices.tolist()
    if avoid_first:
        node_indices = [1+x for x in node_indices]
    return node_indices

def check_connected(graph):
    """
    Given a graph in the form of a list of neighbor lists, check if it is 
    connected.
    """
    n = len(graph)
    reached = [False for _ in range(n)]
    
    def dfs(i, graphx, reachedx):
        for j in graphx[i]:
            if not reachedx[j]:
                reachedx[j] = True
                dfs(j, graphx, reachedx)
    reached[0] = True
    dfs(0, graph, reached)
    return all(reached)

def get_graph(num_nodes, node_min_degree, node_max_degree):
    """
    Generate a random graph with num_nodes nodes without multiple edges or
    self-loops, where each node has a degree in [node_min_degree, 
    node_max_degree].
    """
    while True:
        degrees = np.random.randint(low=node_min_degree, high=node_max_degree, \
            size=num_nodes)
        while np.sum(degrees) % 2 == 1:
            degrees = np.random.randint(low=node_min_degree, high=node_max_degree, \
                size=num_nodes)
        graph = nx.configuration_model(degrees)
        graph.remove_edges_from(nx.selfloop_edges(graph))
        graph = nx.Graph(graph)
        edges = [[] for _ in range(num_nodes)]
        for a,b in list(graph.edges()):
            edges[a].append(b)
            edges[b].append(a)
        degrees = [len(node) for node in edges]
        if any(degree < node_min_degree or degree > node_max_degree for degree \
            in degrees):
            continue
        return edges

def build_nx_graph(graph):
    """
    Convert a graph represented in the form of a list of neighbor lists to
    a networkx graph -- for the purpose of saving as an image.
    """
    edgelist = []
    for i in range(len(graph)):
        for j in graph[i]:
            if j > i:
                edgelist.append((i,j))
    return nx.from_edgelist(edgelist)

def build_graph_with_attacker(num_nodes, node_min_degree=4, node_max_degree=8, \
    save_path=None):
    """
    The attacker always is the first node in this function. The rest is the same
    as the function below.
    """
    num_attacker_neighbors = \
        int(round((num_nodes-1)*HONEST_NODES_CONNECTED_TO_ATTACKER_PCT/100.0))
    while True:
        graph = get_graph(num_nodes-1, node_min_degree, node_max_degree)
        if not check_connected(graph):
            continue
        # Shift the node numbers up
        graph = [[]] + [[a+1 for a in b] for b in graph]
        possiblities = [i for i in range(1,len(graph)) if \
            len(graph[i]) < node_max_degree]
        if len(possiblities) < num_attacker_neighbors:
            continue
        neighs = random.choices(possiblities, k=num_attacker_neighbors)
        graph = [neighs] + graph[1:]
        for i in neighs:
            graph[i].append(0)
        break

    if save_path is not None:
        nx_graph = build_nx_graph(graph)
        plt.clf()
        plt.title("Connectivity Graph")
        nx.draw(nx_graph, with_labels=True)
        plt.savefig(save_path)
    return graph

def build_graph(num_nodes, node_min_degree=4, node_max_degree=8, save_path=None, \
    with_attacker=False):
    """
    Build a random graph with num_nodes nodes that is connected, and where
    each nodes has a degree between node_min_degree and node_max_degree.
    Optionally, save the graph as an image.
    If @with_attacker is set, the node with index 0 is taken as the attacker
    and is handled separately.
    """
    if with_attacker:
        return build_graph_with_attacker(num_nodes, node_min_degree, \
            node_max_degree, save_path)
    graph = get_graph(num_nodes, node_min_degree, node_max_degree)
    while not check_connected(graph):
        graph = get_graph(num_nodes, node_min_degree, node_max_degree)
    if save_path is not None:
        nx_graph = build_nx_graph(graph)
        plt.clf()
        plt.title("Connectivity Graph")
        nx.draw(nx_graph, with_labels=True)
        plt.savefig(save_path)
    return graph

def uniform_random_time(start, stop):
    """
    Uniformly randomly sample a time from [start, stop].
    """
    return random.uniform(start, stop)

def random_traversal(s):
    """
    Shuffle the elements of s for a random iteration order.
    """
    sl = list(s)
    random.shuffle(sl)
    return sl

def get_txn_size_in_coins(coins_in_hand, round_to=TXN_COINS_RESOLUTION):
    """
    Uniformly randomly sample a transaction amount from [MIN_TXN_COINS,
    MAX_TXN_COINS], and round it to the specified resolution.
    """
    txn_size = random.uniform(MIN_TXN_COINS, min(MAX_TXN_COINS, coins_in_hand))
    txn_size = round(txn_size, round_to)
    return txn_size

def get_random_peer_id(num_peers, peer_to_avoid=-1):
    """
    Randomly select a peer from 0, 1, ..., num_peers while optionally avoiding
    peer_to_avoid.
    """
    peer = peer_to_avoid
    while peer == peer_to_avoid:
        peer = random.randint(0, num_peers-1)
    return peer

def get_ticks_to_next_txn(mean_node_time_to_txn_in_secs):
    """
    Sample the time to the generation of the next transaction from an exponential
    distribution with mean mean_node_time_to_txn_in_secs.
    """
    mean_node_time_to_txn_in_ticks = seconds_to_ticks(mean_node_time_to_txn_in_secs)
    ticks = int(np.random.exponential(mean_node_time_to_txn_in_ticks))
    return ticks

def get_ticks_to_next_block(mean_node_iat_in_secs):
    """
    Sample the time to the node's next mining event from an exponential
    distribution with mean mean_node_iat_in_secs.
    """
    mean_node_iat_in_ticks = seconds_to_ticks(mean_node_iat_in_secs)
    ticks = int(np.random.exponential(mean_node_iat_in_ticks))
    return ticks

def get_d_value_in_ticks(c_value, round=False):
    """
    Calculate the value of d_{ij} for message delay and convert it to ticks.
    """
    lambda_value_in_ticks = LAMBDA_FACTOR / c_value
    d_in_ticks = np.random.exponential(lambda_value_in_ticks)
    if round:
        d_in_ticks = int(d_in_ticks)
    return d_in_ticks

def delay_in_ticks(message_size_in_bits, rho_value, c_value):
    """
    Calculate message delay in ticks from node i to node j.
    """
    d_in_ticks = get_d_value_in_ticks(c_value)
    m_by_c_in_ticks = message_size_in_bits / c_value
    delay = int(rho_value + m_by_c_in_ticks + d_in_ticks)
    return delay

def check_close(coins1, coins2):
    """
    In python, floats are not precise - it often represents, e.g. 2 as
    2.0000000000004. Therefore, checking for equality with == may give
    false negatives. This function tolerates differences upto 1e-8, and
    performs checking of equality for a pair of lists.
    """
    return np.allclose(np.array(coins1), np.array(coins2))

if __name__ == '__main__':
    pass