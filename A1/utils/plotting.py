"""
Adithya Bhaskar, 2023.
This file exposes functions to plot tree structures.
"""

from config import *
from utils.globals import *

import os
import pygraphviz

def init_graph(node_id):
    G = pygraphviz.AGraph(directed=True, strict=True)
    G.graph_attr['label'] = 'Blocktree at Node {}'.format(node_id)
    G.node_attr['shape'] = 'box'
    G.node_attr['style'] = 'filled'
    return G

def add_edges(G, blocktree, node_idx):
    for child_idx in blocktree[node_idx].children:
        G.add_edge(blocktree[node_idx].block.block_id, \
            blocktree[child_idx].block.block_id)
        add_edges(G, blocktree, child_idx)
        
def get_blocks_on_path(block):
    nop = [] if block.parent is None else get_blocks_on_path(block.parent)
    nop.append(block.block_id)
    return nop

def add_nodes(G, blocktree, node_idx, blocks_on_longest_chain, node_id):
    block_id = blocktree[node_idx].block.block_id
    miner = blocktree[node_idx].block.miner
    if miner == node_id:
        color = MY_LONG_NODE_COLOR if block_id in blocks_on_longest_chain \
            else MY_NODE_COLOR
    else:
        color = LONG_NODE_COLOR if block_id in blocks_on_longest_chain \
            else NODE_COLOR
    if miner < 0:
        color = GENESIS_COLOR
        label = "Block {} [GENESIS]".format(block_id)
    else:
        label = "Block {} [Node {}]".format(block_id, miner)
    G.add_node(block_id, label=label, color=color)
    
    for child_idx in blocktree[node_idx].children:
        add_nodes(G, blocktree, child_idx, blocks_on_longest_chain, node_id)

def plot_blocktree(node_id, blocktree, longest_block, save_path=None):
    G = init_graph(node_id)
    blocks_on_longest_chain = get_blocks_on_path(longest_block)
    add_nodes(G, blocktree, 0, blocks_on_longest_chain, node_id)
    add_edges(G, blocktree, 0)
    G.layout(prog='dot')
    
    if save_path is not None:
        G.draw(save_path)

if __name__ == '__main__':
    pass