"""
Adithya Bhaskar, 2023.
This file implements a node (a peer) along with their characterestics and
their behavior.
"""

from config import *
from utils.globals import *
from utils.node_utils import *
from utils.plotting import plot_blocktree
from simulator.simulation import *

## Nodes 
NODES = []
## -- Nodes

## Constants for transactions and blocks
COINBASE = -1
TXN_SIZE = TXN_SIZE_IN_KBS * KBIT_SIZE * BYTE_SIZE
MAX_BLOCK_SIZE = MAX_BLOCK_SIZE_IN_MBS * MBIT_SIZE * BYTE_SIZE
NOT_FORWARDED = -12

ERR_BLOCK_FULL = -11
ERR_TXN_INVALID = -12
OK_ADDED_TXN = 0

## -- Constants for transactions and blocks

## Transaction and Block IDs
NUM_TXNS = 0
NUM_BLOCKS = 0

def get_num_txns():
    global NUM_TXNS
    return NUM_TXNS

def inc_and_return_next_txn_id():
    global NUM_TXNS
    NUM_TXNS += 1
    return NUM_TXNS

def get_num_blocks():
    global NUM_BLOCKS
    return NUM_BLOCKS

def inc_and_return_next_block_id():
    global NUM_BLOCKS
    NUM_BLOCKS += 1
    return NUM_BLOCKS

GENESIS_BLOCK = None

## -- Transaction and Block IDs

class Transaction:
    def __init__(self, source_peer, dest_peer, txn_size_in_coins):
        self.source_peer = source_peer
        self.dest_peer = dest_peer
        self.txn_size_in_coins = txn_size_in_coins
        self.size = TXN_SIZE
        self.txn_id = inc_and_return_next_txn_id()
    
    def get_print_name(self):
        if self.source_peer == COINBASE:
            return "{}: {} mines 50 coins".format(self.txn_id, self.dest_peer)
        else:return "{}: {} pays {} {:.4f} coins".format(self.txn_id, \
            self.source_peer, self.dest_peer, self.txn_size_in_coins)
        
    def is_coinbase_txn(self):
        return self.source_peer == COINBASE

class Block:
    def __init__(self, miner, parent=None):
        self.miner = miner
        self.block_id = inc_and_return_next_block_id()
        self.block_size = TXN_SIZE
        self.cumulative_size = self.block_size
        self.transactions = [Transaction(COINBASE, self.miner, MINING_REWARD)] \
            if miner != COINBASE else []
        self.parent = parent
        self.depth = 0 if parent is None else 1 + parent.depth
        self.mine_time = EVENT_NONE_TIME
        self.per_node_balances = [0 for _ in range(NUM_NODES)] if parent \
            is None else parent.per_node_balances.copy()
    
    def set_mine_time(self, mine_time):
        self.mine_time = mine_time
    
    def get_description(self):
        print_string = "Block {}".format(self.block_id)
        for transaction in self.transactions:
            print_string += "\n\t" + transaction.get_print_name()
        return print_string
    
    def validate_transaction(self, transaction):
        return True if transaction.source_peer == COINBASE or \
            self.per_node_balances[transaction.source_peer] \
            >= transaction.txn_size_in_coins else False
    
    def validate_block(self):
        if self.block_id == GENESIS_BLOCK.block_id:
            return False if any(self.per_node_balances) else True
        cur_balances = self.parent.per_node_balances.copy()
        for txn in self.transactions:
            if txn.source_peer != COINBASE and cur_balances[txn.source_peer] \
                < txn.txn_size_in_coins:
                return False
            cur_balances[txn.dest_peer] += txn.txn_size_in_coins
            if txn.source_peer != COINBASE:
                cur_balances[txn.source_peer] -= txn.txn_size_in_coins
        close = check_close(cur_balances, self.per_node_balances)
        return True if close else False
    
    def can_add_txn(self):
        return self.block_size + TXN_SIZE <= MAX_BLOCK_SIZE
        
    def add_txn_to_block(self, transaction):
        if not self.can_add_txn():
            return ERR_BLOCK_FULL
        if not self.validate_transaction(transaction):
            return ERR_TXN_INVALID
        self.transactions.append(transaction)
        if transaction.source_peer != COINBASE:
            self.per_node_balances[transaction.source_peer] -= \
                transaction.txn_size_in_coins
        self.per_node_balances[transaction.dest_peer] += \
            transaction.txn_size_in_coins
        self.block_size += transaction.size
        return OK_ADDED_TXN
    
    def count_blocks_on_path(self, node_id):
        contribution = 1 if self.miner == node_id else 0
        return contribution + (0 if self.parent is None else \
            self.parent.count_blocks_on_path(node_id))
    
    def mining_complete(self):
        # As mining is complete, increase the balance of the miner by the reward
        self.per_node_balances[self.miner] += \
            self.transactions[0].txn_size_in_coins
        self.set_mine_time(get_global_time())
        self.cumulative_size = self.block_size + self.parent.cumulative_size
        NODES[self.miner].num_blocks_mined += 1

class BlockTreeNode:
    def __init__(self, block, parent=None):
        self.block = block
        self.parent = parent
        self.discovery_time = get_global_time()
        self.children = []
        self.forwarded_time = {}
        
    def add_child(self, child):
        self.children.append(child)
        
    def get_forwarded_time(self, neighbor):
        if self.block.block_id == GENESIS_BLOCK.block_id or \
            (neighbor in self.forwarded_time and \
            self.forwarded_time[neighbor] <= get_global_time()):
            return get_global_time()
        elif neighbor in self.forwarded_time:
            return self.forwarded_time[neighbor]
        else:
            return NOT_FORWARDED
        
    def set_forwarded_time(self, neighbor, recv_time):
        self.forwarded_time[neighbor] = recv_time
        
    def forwarding_done_or_scheduled(self, neighbor):
        return True if self.block.block_id == GENESIS_BLOCK.block_id or \
            neighbor in self.forwarded_time else False

class Node:
    def __init__(self, node_id, is_slow, has_low_cpu, total_power):
        self.node_id = node_id
        self.is_slow = is_slow
        self.has_low_cpu = has_low_cpu
        self.inv_hash_power_fraction = total_power if has_low_cpu else \
            total_power/HIGH_TO_LOW_CPU_RATIO
        self.neighbor_ids = []
        self.neighbor_c_values = []
        self.neighbor_latency_rhos = []
        
        self.current_longest_block = GENESIS_BLOCK
        self.block_being_mined = None
        self.current_mining_event = EVENT_NONE_ID
        self.num_blocks_mined = 0
        
        self.currently_included_txns = set()
        self.block_tree = [BlockTreeNode(self.current_longest_block)]
        self.block_to_tree_node_index = {GENESIS_BLOCK.block_id : 0}
        
        self.next_txn_time = None
        self.txns_heard_of = []     # don't care about coinbase ones
        self.txn_id_to_index = {}
        self.txn_indices_not_included = set()
        
        self.logging_disabled = SIMULATION.logging_disabled
        SIMULATION.log_info("Node {} is {} and {}-cpu".format(self.node_id, \
            "slow" if is_slow else "fast", "low" if has_low_cpu else "high"))
        
    def add_neighbor(self, neighbor_id, c_value, rho_value):
        self.neighbor_ids.append(neighbor_id)
        self.neighbor_c_values.append(c_value)
        self.neighbor_latency_rhos.append(rho_value)
        
    def mine_new_block(self, **kwargs):
        mine_time = get_global_time()
        
        # Finish mining, mark included transactions as done
        # No need to re-build helper structures
        self.current_longest_block = self.block_being_mined
        self.current_longest_block.mining_complete()
        
        # Log stuff
        self.log_info("Mined block with ID {}".format( \
            self.current_longest_block.block_id), key="mining")
        self.log_info(self.current_longest_block.get_description(), key="mining")
        self.log_info("Mined block with ID {} at depth {}".format( \
            self.current_longest_block.block_id, \
            self.current_longest_block.depth), key="tree-updates")
        
        self.try_adding_block_to_tree(self.current_longest_block)
        self.update_transactions_included(self.current_longest_block)
        self.block_being_mined = None
        self.current_mining_event = EVENT_NONE_ID
            
        # Broadcast our block
        self.broadcast_block(self.current_longest_block)
        
        # Schedule next mining event
        self.schedule_next_mining_event()
        
    def schedule_next_mining_event(self):
        next_mine_time = get_global_time() + get_ticks_to_next_block( \
            MEAN_INTER_BLOCK_ARRIVAL_TIME_IN_SECS * self.inv_hash_power_fraction)
        
        # Cancel the current mining event if there is one
        if self.current_mining_event != EVENT_NONE_ID:
            SIMULATION.cancel_event(self.current_mining_event)
            self.current_mining_event = None
        
        # Choose transactions to include randomly
        self.block_being_mined = Block(self.node_id, self.current_longest_block)
        for index in random_traversal(self.txn_indices_not_included):
            success = self.block_being_mined.add_txn_to_block( \
                self.txns_heard_of[index])
            if success == ERR_BLOCK_FULL:
                break
        
        # Schedule the mining event
        mining_handler = Node.mine_new_block
        mining_kwargs = { "self" : self }
        mining_event = Event(mining_handler, handler_kwargs=mining_kwargs) 
        SIMULATION.schedule_event_at_time(mining_event, next_mine_time)  
        self.current_mining_event = mining_event.event_number     

    def generate_new_txn(self, **kwargs):
        # Send money only if not broke, but schedule next event nonetheless
        coins_in_hand = self.block_being_mined.per_node_balances[self.node_id]
        if coins_in_hand > MIN_TXN_COINS:
            txn_size_coins = get_txn_size_in_coins(coins_in_hand)
            sending_to = get_random_peer_id(NUM_NODES, self.node_id)
            generated_txn = Transaction(self.node_id, sending_to, txn_size_coins)
            self.txn_indices_not_included.add(len(self.txns_heard_of))
            self.txn_id_to_index[generated_txn.txn_id] = len(self.txns_heard_of)
            self.txns_heard_of.append(generated_txn)
            self.broadcast_txn(generated_txn)
        self.schedule_next_txn_generation_event()
        
    def schedule_next_txn_generation_event(self):
        self.next_txn_time = get_global_time() + get_ticks_to_next_txn( \
            MEAN_PER_NODE_TXN_GENERATION_TIME_IN_SECS)
        
        txn_handler = Node.generate_new_txn
        txn_kwargs = { "self" : self }
        txn_event = Event(txn_handler, handler_kwargs=txn_kwargs)
        SIMULATION.schedule_event_at_time(txn_event, self.next_txn_time)
    
    def receive_txn_message(self, transaction, heard_from, **kwargs):
        if transaction.txn_id in self.txn_id_to_index:
            # Already heard of this and must've forwarded, discard.
            return
        self.log_info("Heard of transaction {} from node {}".\
            format(transaction.txn_id, heard_from), key="gossip")
        self.txn_indices_not_included.add(len(self.txns_heard_of))
        self.txn_id_to_index[transaction.txn_id] = len(self.txns_heard_of)
        self.txns_heard_of.append(transaction)
        self.broadcast_txn(transaction, avoid_node=heard_from)
    
    def broadcast_txn(self, transaction, avoid_node=-1):
        txn_recv_handler = Node.receive_txn_message
        for neighbor, rho, c in zip(self.neighbor_ids, self.neighbor_latency_rhos, \
            self.neighbor_c_values):
            if neighbor == avoid_node:
                continue
            txn_recv_kwargs = {
                "self" : NODES[neighbor], 
                "transaction" : transaction,
                "heard_from" : self.node_id
            }
            txn_recv_event = Event(txn_recv_handler, handler_kwargs=txn_recv_kwargs)
            txn_recv_time = get_global_time() + delay_in_ticks(transaction.size, \
                rho, c)
            SIMULATION.schedule_event_at_time(txn_recv_event, txn_recv_time)

    def try_adding_block_to_tree(self, block):
        if block.block_id in self.block_to_tree_node_index:
            # We know of this block, so we must have validated it
            return block.depth
        parent_depth = self.try_adding_block_to_tree(block.parent)
        if parent_depth+1 != block.depth:
            # We could only partially add our chain due to verification failing
            # Abort all the way to the leaf
            return parent_depth
        # We always return the maximum depth until which verification succeeded
        if not block.validate_block():
            return parent_depth
        parent_index = self.block_to_tree_node_index[block.parent.block_id]
        new_index = len(self.block_tree)
        self.block_tree.append(BlockTreeNode(block, parent_index))
        self.block_tree[parent_index].add_child(new_index)
        self.block_to_tree_node_index[block.block_id] = new_index
        
        self.log_info("Added block {} to tree at depth {}".format( \
            block.block_id, block.depth), key="tree-updates")
        return block.depth
    
    def update_transactions_included(self, block, recursive=False):
        if block.block_id == GENESIS_BLOCK.block_id:
            return
        # We do not care about coinbase transactions
        for txn in block.transactions[1:]:
            if txn.txn_id in self.txn_id_to_index:
                txn_index = self.txn_id_to_index[txn.txn_id]
                # discard = remove if exists else no-op
                self.txn_indices_not_included.discard(txn_index)
            else:
                # Don't add to txn_indices_not_included
                self.txn_id_to_index[txn.txn_id] = len(self.txns_heard_of)
                self.txns_heard_of.append(txn)
        if recursive:
            self.update_transactions_included(block.parent, recursive)
    
    def update_tree_with_new_longest_block(self, block):
        if block.depth <= self.current_longest_block.depth:
            return
        
        # We need to rebuild our set of transactions not included
        # The set of transactions we've heard of may also be updated
        self.txn_indices_not_included = set(range(len(self.txns_heard_of)))
        self.update_transactions_included(block, recursive=True)
        
        self.current_longest_block = block
        
        self.log_info("Switching over to block {} at depth {} as longest chain".\
            format(block.block_id, block.depth), key="tree-updates")
        
        # Our mining efforts were futile, retry.
        self.schedule_next_mining_event()
    
    def receive_block_message(self, block, heard_from, **kwargs):
        if block.block_id in self.block_to_tree_node_index:
            # Already know of this block
            return
        self.log_info("Heard of block {} from node {}".\
            format(block.block_id, heard_from), key="gossip")
        
        mine_to_recv_time = ticks_to_seconds(get_global_time() - block.mine_time)
        metric_name = "Mine-to-Discovery for {}-{} nodes".format(\
            "slow" if self.is_slow else "fast", \
            "lowcpu" if self.has_low_cpu else "highcpu")
        SIMULATION.log_aggregate_metric(metric_name, mine_to_recv_time)
        
        # Add the extra blocks to our tree (maybe only partially)
        max_depth = self.try_adding_block_to_tree(block)
        if max_depth > self.current_longest_block.depth:
            # Found a new deepest block - switch over
            self.update_tree_with_new_longest_block(block)
            
        self.broadcast_block(block, avoid_node=heard_from)
        
    def broadcast_block(self, block, avoid_node=-1):
        bnode = self.block_to_tree_node_index[block.block_id]
        for neighbor, rho, c in zip(self.neighbor_ids, self.neighbor_latency_rhos, \
            self.neighbor_c_values):
            if neighbor == avoid_node:
                continue
            self.send_block_to_neighbor_with_corner_cases(bnode, neighbor, rho, c)
    
    def send_block_to_neighbor_with_corner_cases(self, bnode, neighbor, rho, c):
        if self.block_tree[bnode].forwarding_done_or_scheduled(neighbor):
            return self.block_tree[bnode].get_forwarded_time(neighbor)
        parent = self.block_tree[bnode].parent
        forwarded_time = self.block_tree[parent].get_forwarded_time(neighbor)
        if forwarded_time == NOT_FORWARDED:
            forwarded_time = self.send_block_to_neighbor_with_corner_cases(\
                parent, neighbor, rho, c)
        recv_time = self.send_block_to_neighbor(self.block_tree[bnode].block, \
            neighbor, rho, c, start_time=forwarded_time)
        self.block_tree[bnode].set_forwarded_time(neighbor, recv_time)
        return recv_time
    
    def send_block_to_neighbor(self, block, neighbor, rho, c, \
        start_time=get_global_time()):
        block_recv_kwargs = {
            "self" : NODES[neighbor], 
            "block" : block,
            "heard_from" : self.node_id
        }
        block_recv_handler = Node.receive_block_message
        block_recv_event = Event(block_recv_handler, \
            handler_kwargs=block_recv_kwargs)
        msg_size = block.cumulative_size if \
            SIMULATE_FORWARDING_OF_ENTIRE_CHAINS else block.block_size
        delay = delay_in_ticks(msg_size, rho, c)
        block_recv_time = start_time + delay
        SIMULATION.schedule_event_at_time(block_recv_event, block_recv_time)
        return block_recv_time
    
    def log_info(self, log_message, key="general"):
        SIMULATION.log_node(self.node_id, log_message, key=key)
        
    def finish_and_save_tree(self, node_dir):
        if not os.path.exists(node_dir):
            os.makedirs(node_dir, exist_ok=True)
        out_path = os.path.join(node_dir, "tree.out")
        output = ["*** Blocktree for node {}\n".format(self.node_id)]
        for blocknode in self.block_tree:
            block = blocknode.block
            output.append("\n* Block {} *".format(block.block_id))
            output.append("Depth {}".format(block.depth))
            if block.miner == COINBASE:
                output.append("Mined by COINBASE [GENESIS block]")
            else:
                output.append("Mined by node {} at {} seconds.".format(\
                    block.miner, ticks_to_seconds(block.mine_time)))
            output.append("Discovered at: {} seconds".format(ticks_to_seconds( \
                blocknode.discovery_time)))
            if block.parent is None:
                output.append("Parent is NONE [Root]")
            else:
                output.append("Parent is Block {}".format(block.parent.block_id))
            if len(blocknode.children) == 0:
                output.append("No children")
            else:
                children_string = "Children Block IDs: "
                for child in blocknode.children:
                    children_string += " {}".format(\
                        self.block_tree[child].block.block_id)
                output.append(children_string)
            output.append("Block description: ")
            output.append(block.get_description())
            
        num_blocks_on_path = self.current_longest_block.count_blocks_on_path(\
            self.node_id)
        metric_name = "% of node {}'s blocks on its longest chain".format(\
            self.node_id)
        
        if self.num_blocks_mined > 0:
            ptage = 100.0*num_blocks_on_path/self.num_blocks_mined 
            output.append("\n% of node {}'s blocks on its longest chain : {:.4f}%".\
                format(self.node_id, ptage))
            metric_name = "%age of mined blocks on their longest chain,"+\
                " {}-{} nodes".format("slow" if self.is_slow else "fast", \
                "lowcpu" if self.has_low_cpu else "highcpu")
            SIMULATION.log_aggregate_metric(metric_name, ptage)
        else:
            output.append("\nNo blocks mined.")
        
        with open(out_path, 'w+') as f:
            f.write("\n".join(output))
         
        ptage = 100.0*num_blocks_on_path/(1+self.current_longest_block.depth)
        metric_name = "%age of blocks on their longest chain that's theirs,"+\
            " {}-{} nodes".format("slow" if self.is_slow else "fast", \
            "lowcpu" if self.has_low_cpu else "highcpu")
        SIMULATION.log_aggregate_metric(metric_name, ptage)        
        
        tree_png_out = os.path.join(node_dir, "blocktree.png")
        plot_blocktree(self.node_id, self.block_tree, \
            self.current_longest_block, tree_png_out)    

def finish_and_save_tree_for_all_nodes(log_dir):
    for n in range(NUM_NODES):
        NODES[n].finish_and_save_tree(os.path.join(log_dir, \
            "node{}".format(n)))

def init_nodes():
    """
    Initialize nodes and global variables as follows:
    - Decide which nodes have low computational power and/or are slow.
    - Generate the connectivity graph and save it.
    - Schedule the first mining and transaction generation events.
    - Instantiate the final callback.
    """
    global NODES, GENESIS_BLOCK
    GENESIS_BLOCK = Block(COINBASE)
    slow_node_indices = set(get_node_indices_with_property(NUM_NODES, \
        SLOW_NODES_PERCENT))
    low_cpu_node_indices = set(get_node_indices_with_property(NUM_NODES, \
        LOW_CPU_NODES_PERCENT))
    
    num_low_cpu_nodes = len(low_cpu_node_indices)
    total_power = num_low_cpu_nodes + HIGH_TO_LOW_CPU_RATIO*(NUM_NODES-\
        num_low_cpu_nodes)
    
    for i in range(NUM_NODES):
        NODES.append(Node(i, i in slow_node_indices, i in low_cpu_node_indices, \
            total_power))

    graph_save_path = None if SIMULATION.logging_disabled else \
        os.path.join(SIMULATION.log_dir, "graph-nodes.png")
    edges = build_graph(NUM_NODES, save_path=graph_save_path)

    for i in range(NUM_NODES):
        NODES[i].schedule_next_mining_event()
        NODES[i].schedule_next_txn_generation_event()
        for j in edges[i]:
            c_value = (5*MBPS) if NODES[i].is_slow or NODES[j].is_slow else \
                (100*MBPS)
            rho_value = uniform_random_time(10*MILLISECOND_IN_TICKS, \
                500*MILLISECOND_IN_TICKS)
            NODES[i].add_neighbor(j, c_value, rho_value)
            
    SIMULATION.set_final_callback(finish_and_save_tree_for_all_nodes)

if __name__ == '__main__':
    pass