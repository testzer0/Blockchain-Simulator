import json
import os
from web3 import Web3
import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt

from networkx.generators.degree_seq import random_degree_sequence_graph
from networkx.algorithms.graphical import is_graphical
from networkx.utils.random_sequence import powerlaw_sequence

from collections import deque

## Config 

NUM_NODES = 100
EACH_BALANCE_MEAN = 10
NUM_TXNS = 1000
LOG_EVERY = 100
SEND_AMOUNT = 1
USER_IDS = []
USER_NAMES = []
USER_LOG = []
GRAPH_LOG = []
TXN_LOG = []
RESULT_LOG = []

CONTRACT_ADDRESS = '0xa9208D2Ba6697AaaBBb3DeBeA2f25B0CE9AE0D09'

## -- Config

def get_output_dir_for_run(root_dir="out/"):
    """
    Find out the next output directory - goes in the order log1, log2, ...
    """
    max_existing = 0
    for d in os.listdir(root_dir):
        if d.startswith("run") and d[3:].isnumeric():
            max_existing = max(max_existing, int(d[3:]))
    return os.path.join(root_dir, "run{}".format(max_existing+1))

def save_graph(graph, save_path="out/graph.png"):
    """
    Save an image of @graph to @save_path.
    """
    edgelist = []
    for i in range(len(graph)):
        for j in graph[i]:
            if j > i:
                edgelist.append((i,j))
    g = nx.from_edgelist(edgelist)
    labeldict = {i: str(i+1) for i in range(len(graph))}
    plt.clf()
    plt.title("Connectivity Graph")
    nx.draw(g, with_labels=True, labels=labeldict)
    plt.savefig(save_path)

def get_random_powerlaw_graph(num_nodes=NUM_NODES):
    """
    Construct a random graph of @num_nodes nodes with the power-law degree
    distribution, and return its adjacency matrix.
    """
    while True:
        degrees = np.array([int(round(d)) for d in powerlaw_sequence(num_nodes)])
        if not is_graphical(degrees):
            continue
        graph = nx.configuration_model(degrees)
        graph.remove_edges_from(nx.selfloop_edges(graph))
        graph = nx.Graph(graph)
        if not nx.is_connected(graph):
            continue
        edges = [[] for _ in range(num_nodes)]
        for a,b in list(graph.edges()):
            edges[a].append(b)
            edges[b].append(a)
        return edges

def get_amount_value(mean=EACH_BALANCE_MEAN, round=True):
    """
    Sample a random amount value from the exponential distribution with mean
    @mean. By default, also rounds the value to the nearest integer.
    """
    value = np.random.exponential(mean)
    if round:
        value = int(value)
    return value

def get_shortest_path(from_, to_, graph):
    """
    Calculate the shortest path from @from_ to @to_ in @graph, and return it
    as a list.
    """
    n = len(graph)
    distances = [n for _ in range(n)]
    nexthop = [n for _ in range(n)]
    distances[to_] = 0
    queue = deque()
    queue.append(to_)
    while len(queue) > 0:
        source = queue.popleft()
        for neigh in graph[source]:
            if distances[neigh] > distances[source] + 1:
                distances[neigh] = distances[source] + 1
                nexthop[neigh] = source
                queue.append(neigh)
    assert(distances[from_] < n)
    path = []
    cur = from_
    while cur != to_:
        path.append(cur)
        cur = nexthop[cur]
    path.append(to_)
    return path

GRAPH = get_random_powerlaw_graph(NUM_NODES)
OUT_DIR = get_output_dir_for_run()
os.makedirs(OUT_DIR, exist_ok=True)
save_graph(GRAPH, save_path=os.path.join(OUT_DIR, "graph.png"))

#connect to the local ethereum blockchain
provider = Web3.HTTPProvider('http://127.0.0.1:8545')
w3 = Web3(provider)
#check if ethereum is connected
assert(w3.is_connected())

#replace the address with your contract address (!very important)
deployed_contract_address = CONTRACT_ADDRESS

#path of the contract json file. edit it with your contract json file
compiled_contract_path ="build/contracts/Payment.json"
with open(compiled_contract_path) as file:
    contract_json = json.load(file)
    contract_abi = contract_json['abi']
contract = w3.eth.contract(address = deployed_contract_address, abi = contract_abi)



'''
#Calling a contract function createAcc(uint,uint,uint)
txn_receipt = contract.functions.createAcc(1, 2, 5).transact({'txType':"0x3", 'from':w3.eth.accounts[0], 'gas':2409638})
txn_receipt_json = json.loads(w3.to_json(txn_receipt))
print(txn_receipt_json) # print transaction hash

# print block info that has the transaction)
print(w3.eth.get_transaction(txn_receipt_json)) 

#Call a read only contract function by replacing transact() with call()

'''

#Add your Code here

txn_hash = contract.functions.resetAll().transact({'txType':"0x3", \
    'from':w3.eth.accounts[0], 'gas':2409638})
txn_receipt_json = w3.eth.wait_for_transaction_receipt(txn_hash)

USER_LOG.append("*** Users ***")
USER_LOG.append("User ID\t\tUsername\n")
for i in range(NUM_NODES):
    # Register each user
    USER_IDS.append(i+1)
    USER_NAMES.append("USER {}".format(i+1))
    USER_LOG.append("{}\t\t{}".format(USER_IDS[i], USER_NAMES[i]))
    txn_hash = contract.functions.registerUser(USER_IDS[i], USER_NAMES[i]).\
        transact({'txType':"0x3", 'from':w3.eth.accounts[0], 'gas':2409638})
    txn_receipt_json = w3.eth.wait_for_transaction_receipt(txn_hash)
    result = contract.functions.getLastSuccessCode().call()
    assert(result)

GRAPH_LOG.append("*** Connectivity Graph ***")
GRAPH_LOG.append("User ID 1\t\tUser ID 2\t\tInitial Contribution Each\n")
for i in range(len(GRAPH)):
    for j in GRAPH[i]:
        if i >= j:
            # Do not add the same account twice
            continue
        # Add each account
        amount = get_amount_value()
        GRAPH_LOG.append("{}\t\t{}\t\t{}".format(USER_IDS[i], USER_IDS[j], amount))
        txn_hash = contract.functions.createAccount(USER_IDS[i], USER_IDS[j], \
            amount).transact({'txType':"0x3", 'from': w3.eth.accounts[0], \
            'gas':2409638})
        txn_receipt_json = w3.eth.wait_for_transaction_receipt(txn_hash)
        result = contract.functions.getLastSuccessCode().call()
        assert(result)

TXN_LOG.append("*** Transaction Log ***")
TXN_LOG.append("ID From\t\tID To\t\tAmount\t\tSuccess?\n")
n_txns_tried = 0
n_txns_succeeded = 0

x_axis = []
y_axis = []

RESULT_LOG.append("*** Results ***")
RESULT_LOG.append("View success_rate.png for the corresponding plot.\n")

for _ in range(NUM_TXNS):
    # Try firing a random transaction
    i = random.randint(0, NUM_NODES-1)
    j = i 
    while j == i:
        j = random.randint(0, NUM_NODES-1)
    path = get_shortest_path(i, j, GRAPH)
    id_path = [USER_IDS[x] for x in path]
    txn_hash = contract.functions.sendAmount(id_path, SEND_AMOUNT).transact(\
        {'txType':"0x3", 'from':w3.eth.accounts[0], 'gas':2409638})
    txn_receipt_json = w3.eth.wait_for_transaction_receipt(txn_hash)
    result = contract.functions.getLastSuccessCode().call()
    TXN_LOG.append("{}\t\t\t{}\t\t\t{}\t\t\t{}".format(USER_IDS[i], USER_IDS[j], \
        SEND_AMOUNT, str(result)))
    n_txns_tried += 1
    n_txns_succeeded += int(result)
    
    if n_txns_tried % LOG_EVERY == 0:
        x_axis.append(n_txns_tried)
        success = 100.0*n_txns_succeeded/n_txns_tried
        y_axis.append(success)
        RESULT_LOG.append("After {} transactions: {} succeeded".format(n_txns_tried, \
            n_txns_succeeded) + " (success rate = {:.2f}%)".format(success))
        
with open(os.path.join(OUT_DIR, "users.txt"), 'w+') as f:
    f.write("\n".join(USER_LOG))
    
with open(os.path.join(OUT_DIR, "graph.txt"), 'w+') as f:
    f.write("\n".join(GRAPH_LOG))
    
with open(os.path.join(OUT_DIR, "transactions.log"), 'w+') as f:
    f.write("\n".join(TXN_LOG))
    
with open(os.path.join(OUT_DIR, "results.txt"), 'w+') as f:
    f.write("\n".join(RESULT_LOG))

# Save the plot    
plt.clf()
plt.plot(x_axis, y_axis, label="Success Rate (%)")
plt.title("Success Rate v/s No. of TXNs")
plt.xlabel("No. of Transactions tried")
plt.ylabel("Success Rate (%)")
plt.savefig(os.path.join(OUT_DIR, "success_rate.png"))