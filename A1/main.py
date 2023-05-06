"""
Adithya Bhaskar, 2023.
This file contains the code that brings everything together and runs \
the simulation.
"""

from config import *
from utils.globals import *

from simulator.simulation import *
from simulator.nodes import *

def serve():
    init_nodes()
    SIMULATION.run_simulation()

if __name__ == '__main__':
    serve()