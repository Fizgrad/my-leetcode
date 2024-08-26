
#include <iostream>
#include <vector>

using namespace std;

// Definition for a Node.
class Node {
public:
    int val;
    vector<Node *> children;

    Node() {}

    Node(int _val) {
        val = _val;
    }

    Node(int _val, vector<Node *> _children) {
        val = _val;
        children = _children;
    }
};


class Solution {
public:
    vector<int> postorder(Node *root) {
        vector<int> res;
        auto dfs = [&](auto &&dfs, Node *root) {
            if (root == nullptr)
                return;
            for (auto &i: root->children) {
                dfs(dfs, i);
            }
            res.push_back(root->val);
        };
        dfs(dfs, root);
        return res;
    }
};