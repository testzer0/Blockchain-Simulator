import random
import sys
import os

def metrics(in_path, node_num=0):
    parent_map = {}
    depth_map = {}
    miner_map = {}
    
    lines = open(in_path).read().split('\n')
    lines = [line.strip() for line in lines if line.strip() != '']
    max_depth = 0
    for i in range(len(lines)):
        if lines[i].startswith('* Block'):
            block_id = int(lines[i][7:-1].strip())
            depth = int(lines[i+1].strip().split()[-1])
            if block_id == 1:
                miner = -1
                parent = -1
            else:
                miner = int(lines[i+2].strip().split()[3])
                parent = int(lines[i+4].strip().split()[-1])
            depth_map[block_id] = depth
            miner_map[block_id] = miner
            parent_map[block_id] = parent
            max_depth = max(depth, max_depth)
    last_blocks = [block_id for block_id in depth_map if depth_map[block_id] \
        == max_depth]
    my_blocks = [block_id for block_id in last_blocks if miner_map[block_id] \
        == node_num]
    if len(my_blocks) != 0:
        chosen_block = random.choice(my_blocks)
    else:
        chosen_block = random.choice(last_blocks)
    num_blocks_mined_by_adv = len([block_id for block_id in miner_map if \
        miner_map[block_id] == node_num])
    total_blocks = len(parent_map)
    num_blocks_on_main_chain_by_adv = 0
    num_main_chain_blocks = 0
    while chosen_block in parent_map:
        # Not genesis
        num_main_chain_blocks += 1
        if miner_map[chosen_block] == node_num:
            num_blocks_on_main_chain_by_adv += 1
        chosen_block = parent_map[chosen_block]
    print("Without considering GENESIS block:")
    print("Total number of blocks = {}".format(total_blocks))
    print("Total number of blocks on main chain = {}".format(num_main_chain_blocks))
    print("Number of blocks by adversary = {}".format(num_blocks_mined_by_adv))
    print("Number of blocks by adversary on main chain = {}".format( \
        num_blocks_on_main_chain_by_adv))
    mpu_adv = num_blocks_on_main_chain_by_adv / num_blocks_mined_by_adv
    mpu_overall = num_main_chain_blocks / total_blocks
    revenue_ratio = num_blocks_on_main_chain_by_adv / num_main_chain_blocks
    print("MPU (Adversary) = {:.4f}".format(mpu_adv))
    print("MPU (Overall) = {:.4f}".format(mpu_overall))
    print("Revenue Ratio (Adversary) = {:.4f}".format(revenue_ratio))

log_dir = sys.argv[1] if len(sys.argv) > 1 else "logs/run1/"    
path = log_dir if log_dir.endswith("tree.out") else os.path.join(log_dir, \
    "node0", "tree.out")
metrics(path)