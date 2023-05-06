"""
Adithya Bhaskar, 2023.
This file contains the configuration for running the simulation.
"""

from utils.globals import seconds_to_ticks, KBIT_SIZE, MBIT_SIZE, GBIT_SIZE

## Simulation parameters

SIM_START_SECONDS = 0                           # Start time of simulation
SIM_END_SECONDS = 100000                        # End time of simulation

SIM_START = seconds_to_ticks(SIM_START_SECONDS)
SIM_END = seconds_to_ticks(SIM_END_SECONDS)

DETAILED_EVENT_INFO_SIM = False                 # Useless for now, but useful
                                                # to set triggers, etc.
                                                
SIMULATE_FORWARDING_OF_ENTIRE_CHAINS = False    # Typically we will not just
                                                # gossip about blocks but the
                                                # entire chain from root to
                                                # the block - this is because
                                                # the peer may not have the
                                                # block's parent. This decides
                                                # if that influences |m| by
                                                # using the cumulative chain size
                                                # instead of block size

SIM_SELFISH_MINER = False                        # One of the miners is selfish
SIM_STUBBORN_MINER = True                      # One of the miners is stubborn
HONEST_NODES_CONNECTED_TO_ATTACKER_PCT = 50     # Percentage of honest nodes
                                                # connected to attacker
ATTACKER_MINING_POWER_PCT = 40                  # Mining power of attacker (%)
VERBOSITY = 2                                   # Log verbosity [1-4]

## -- Simulation parameters

## Node configurations

NUM_NODES = 100                                 # Number of peers/nodes
SLOW_NODES_PERCENT = 50                         # Percent of slow nodes
LOW_CPU_NODES_PERCENT = 50                      # Percent of nodes with low cpu
HIGH_TO_LOW_CPU_RATIO = 10                      # Mining power ration High/Low
MEAN_INTER_BLOCK_ARRIVAL_TIME_IN_SECS = 600     # Mean block inter-arrival time
MEAN_PER_NODE_TXN_GENERATION_TIME_IN_SECS = 45  # Mean txn-generation time per node
LAMBDA_FACTOR = 96*KBIT_SIZE                    # Numerator for d_{ij} calculation

TXN_SIZE_IN_KBS = 1                             # Transaction size
MAX_BLOCK_SIZE_IN_MBS = 1                       # Upper limit on block size
MIN_TXN_COINS = 0.1                             # Minimum transaction amount
MAX_TXN_COINS = 1                               # Maximum transaction amount
TXN_COINS_RESOLUTION = 4                        # Resolution (decimal places) 
                                                # for transaction amount
MINING_REWARD = 50                              # Mining reward in coins

## -- Node configurations

## Graphing

NODE_COLOR = 'orange'                           # Normal color for block
LONG_NODE_COLOR = 'green'                       # Color of blocks on longest chain
MY_NODE_COLOR = 'red'                           # Color of node's blocks not
                                                # on the longest chain
MY_LONG_NODE_COLOR = 'purple'                   # Color of node's blocks on
                                                # the longest chain
GENESIS_COLOR = 'grey'                          # Color of genesis block 

##

if __name__ == '__main__':
    pass