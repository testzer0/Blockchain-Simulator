# CS 765 Assignment 2

This repository houses the code for Assignment 2 of CS 765 (Spring 2023 offering). The control flow is highlighted in the flowchart in `flowchart.png`. Our report is named `report.pdf`.

## Team Members

- Adithya Bhaskar (190050005)
- Danish Angural (190050028)
- Dev Desai (190020038)

## Environment setup

Run
```
pip3 install -r requirements.txt
```
to install the relevant packages.

## How to run

To run a simulation, simply type

```
python3 main.py
```

after modifying the configuration parameters in `config.py` to your liking. To calculate the metrics for a run, call `metrics.py` with the log directory's path as the only argument:

```
python3 metrics.py logs/run1/
``` 

## Logging information

By default, logs will be stored in `logs/run1`', `logs/run2`, ... . Within a log folder, 
- `simulation.log` logs global information and final aggregated metrics. 
- the connectivity graph is saved in `graph-nodes.png`. 
- node-specific information can be found under the directory `nodeN` for a node with number `N`. Specifically, `tree.out` stores the blocktree while `blocktree.png` visualizes it. Here, the blocktree includes both the main blockchain and orphaned branches. 
- transaction and Block gossip is logged under `gossip.log`. 
- addition of blocks and update of longest chains is logged in `tree-updates.log`.

---
**NOTE**

Each 100-node run generates log information up to the order of 500 MB - 1 GB when at verbosity level 3 or more. In addition, runs with 10 nodes are recommended for quick results (~2 min) whereas the 100 node runs take 40-60 minutes. We recommend setting the verbosity to 2 (default), whereupon transaction gossip is not logged, saving a lot of space. We also suggest using a transaction generation mean interval of ~40s instead of 10 ms as the latter leads to thrashing of RAM, slowing the script down due to page swapping. We observed runtimes of around 8 hours with a 10 ms interval.

---

Considering the large size of the log files, the logs of all the reported runs are hosted on [Google Drive](https://drive.google.com/file/d/167krps9rqVCWx7jiDeuj2a2unkqnjlRc/view?usp=sharing).

The runs from the previous assignment are available [here](https://drive.google.com/file/d/1V8JCH2OhilyPHaGZi-9yVnRMq9cAPupE/view?usp=sharing).