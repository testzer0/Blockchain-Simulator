// SPDX-License-Identifier: MIT
pragma solidity >=0.4.22 <0.9.0;

contract Payment {
    uint constant MAX_USERS = 200;
    uint constant MAX_JOINT_ACCOUNTS = 2000;
    uint num_users;
    uint num_joint_accounts;

    struct User {
        uint user_id;
        string user_name;
    }

    struct JointAccount {
        // The lower user_idx is assigned position 1
        uint contribution_1;
        uint contribution_2;
    }

    // Allows us to see if our last request succeeded
    bool last_success_code;

    // Using non-dynamic arrays is much cheaper 
    User[MAX_USERS] users;
    JointAccount[MAX_JOINT_ACCOUNTS] joint_accounts;
    mapping(uint => uint) user_id_to_idx;
    mapping(uint => uint) user_idxes_to_account;

    function addJointAcount(uint user_idx_1, uint user_idx_2, 
        uint amount) private returns (bool) {
        // Add a joint account between users with indices @user_idx_1 and 
        // @user_idx_2, with initial contribution @amount each. 
        if (num_joint_accounts >= MAX_JOINT_ACCOUNTS) {
            return false;
        }
        if (user_idx_1 > user_idx_2) {
            uint tmp = user_idx_2;
            user_idx_2 = user_idx_1;
            user_idx_1 = tmp;
        }
        uint hash_value = MAX_USERS*(user_idx_1) + user_idx_2;
        user_idxes_to_account[hash_value] = num_joint_accounts;
        joint_accounts[num_joint_accounts] = JointAccount(amount, amount);
        num_joint_accounts++;

        return true;
    }

    function getJointAccountIdx(uint user_idx_1, uint user_idx_2) private view
        returns (uint) {
        // Map the two user indices to the index for their joint account
        if (user_idx_1 > user_idx_2) {
            uint tmp = user_idx_2;
            user_idx_2 = user_idx_1;
            user_idx_1 = tmp;
        }
        uint hash_value = MAX_USERS*(user_idx_1) + user_idx_2;
        return user_idxes_to_account[hash_value];
    }

    function jointAccountHasEnoughFunds(uint user_idx_1, uint user_idx_2, 
        uint amount) private view returns (bool) {
        // Check if the joint account between users with indices 
        // @user_idx_1 and @user_idx_2 has sufficient funds for the former
        // to transfer @amount coins to the latter.
        uint idx;
        idx = getJointAccountIdx(user_idx_1, user_idx_2);
        if (user_idx_1 > user_idx_2) {
            return joint_accounts[idx].contribution_2 >= amount;
        } else {
            return joint_accounts[idx].contribution_1 >= amount;
        }
    }

    function transferFundsInJointAccount(uint user_idx_1, uint user_idx_2, 
        uint amount) private returns (bool) {
        // Transfer @amount coins from the user with index @user_idx_1 to 
        // the user with index @user_idx_2 in their joint account.
        uint idx;
        idx = getJointAccountIdx(user_idx_1, user_idx_2);
        if (user_idx_1 > user_idx_2) {
            if (joint_accounts[idx].contribution_2 < amount) {
                return false;
            } else {
                joint_accounts[idx].contribution_1 += amount;
                joint_accounts[idx].contribution_2 -= amount;
                return true;
            }
        } else {
            if (joint_accounts[idx].contribution_1 < amount) {
                return false;
            } else {
                joint_accounts[idx].contribution_2 += amount;
                joint_accounts[idx].contribution_1 -= amount;
                return true;
            }
        }
    }

    function transferFundsAlongPath(uint[] calldata id_list, uint amount) private 
        returns (bool) {
        // Transfer @amount coins from one user to the next on the path defined
        // by the sequence of user IDs @id_list.
        for (uint i = 0; i+1 < id_list.length; i++) {
            uint cur = user_id_to_idx[id_list[i]];
            uint nxt = user_id_to_idx[id_list[i+1]];
            if (!jointAccountHasEnoughFunds(cur, nxt, amount)) {
                return false;
            }
            cur = nxt;
        }
        for (uint i = 0; i+1 < id_list.length; i++) {
            uint cur = user_id_to_idx[id_list[i]];
            uint nxt = user_id_to_idx[id_list[i+1]];
            bool xfr_success = transferFundsInJointAccount(cur, nxt, amount);
            require(xfr_success);
            cur = nxt;
        }
        return true;
    }

    function registerUser(uint user_id, string memory user_name) public {
        // Register a new user with user ID @user_id and username @user_name.
        if (num_users >= MAX_USERS) {
            last_success_code = false;
        } else {
            users[num_users] = User(user_id, user_name);
            user_id_to_idx[user_id] = num_users;
            num_users++;
            last_success_code = true;
        }
    }

    function createAccount(uint user_id_1, uint user_id_2, uint amount) public {
        // Add a joint account between users with IDs @user_id_1 and @user_id_2,
        // with initial contribution @amount each.
        uint user_idx_1 = user_id_to_idx[user_id_1];
        uint user_idx_2 = user_id_to_idx[user_id_2];

        last_success_code = addJointAcount(user_idx_1, user_idx_2, amount);
    }

    function sendAmount(uint[] calldata id_path, uint amount) public {
        // Transfer @amount coins from one user to the next on the path defined
        // by the sequence of user IDs @id_list.
        last_success_code = transferFundsAlongPath(id_path, amount);
    }

    function closeAccount(uint user_id_1, uint user_id_2) public {
        // Instead of actually closing the account, just set the balances to 0.
        uint user_idx_1 = user_id_to_idx[user_id_1];
        uint user_idx_2 = user_id_to_idx[user_id_2];
        uint account_idx = getJointAccountIdx(user_idx_1, user_idx_2);
        joint_accounts[account_idx].contribution_1 = 0;
        joint_accounts[account_idx].contribution_2 = 0;
        last_success_code = true;
    }

    function getLastSuccessCode() public view returns (bool) {
        // Check if our last action succeeded
        return last_success_code;
    }

    function resetAll() public {
        // Reset everything to the initial state, i.e. no users and no accounts.
        num_users = 0;
        num_joint_accounts = 0;
        last_success_code = false;
    }

    constructor() {
        num_users = 0;
        num_joint_accounts = 0;
        last_success_code = false;
    }
}
