# CS765 Assignment 3

This repository contains the code for Assignment 3 of CS765 at IIT Bombay (2023 offering). The corresponding report can be found in `report.pdf`.

## Team Members

- Adithya Bhaskar (190050005)
- Danish Angural (190050028)
- Dev Desai (190020038)

## Environment setup

Run
```
pip3 install -r requirements.txt
```
to install the relevant python packages. In addition, you should have installed [NodeJS](https://nodejs.org/en/download/package-manager), [Ganache](https://trufflesuite.com/docs/ganache/quickstart/) or [ganache-cli](https://www.npmjs.com/package/ganache-cli), and [Truffle](https://trufflesuite.com/docs/truffle/how-to/install/).


## Running the script
You can modify the configuration paramaters in `client.py` to your liking. Then, follow these steps:
- In a separate terminal, run `ganache-cli`, or Ganache if using the GUI version. In the latter case, you may have to comment out lines 67-71 of `truffle-config.js` first. 
- Next, run 
  ```
  truffle compile
  ```
  and then 
  ```
  truffle migrate
  ```
  in the original terminal. Use the address in the `contract address` field from the output to replace the configuration parameter `CONTRACT_ADDRESS` in line 29 of `client.py`.
- Finally, run 
  ```
  python3 client.py
  ```
  to run the client. The output will be stored in the format specified below. We observed runtimes of around 90 seconds for this step.

## Output location and format
The log-directories are located under the `out/` directory and are called `log1`, `log2`, and so on. A new run creates the corresponding log directory according to this sequence. Within it, 
- user IDs and usernames are saved in `users.txt`.
- A log of the joint accounts and initial balances are given in `graph.txt`, and an image of the connectivity graph is stored in `graph.png`.
- All transactions (both succesful and failed) are logged in `transactions.log` with their user IDs, amounts and status.
- The number of successful transactions and the success percentage is logged versus the number of attempts in `results.txt`. The latter is also plotted as a line graph in `success_rate.png`.