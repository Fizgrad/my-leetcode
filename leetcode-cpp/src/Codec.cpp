#include<iostream>
#include <queue>

using namespace std;

//Definition for a binary tree node.
struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;

    TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};


class Codec {
public:
    // Encodes a tree to a single string.
    string serialize(TreeNode *root) {
        string res = "[";
        queue<TreeNode *> q;
        q.push(root);
        while (!q.empty()) {
            auto i = q.front();
            q.pop();
            if (i == nullptr) {
                res.append("n,");
            } else {
                res.append(to_string(i->val));
                res.append(",");
                q.push(i->left);
                q.push(i->right);
            }
        }
        res.pop_back();
        res.push_back(']');
        return std::move(res);
    }

    // Decodes your encoded data to tree.
    TreeNode *deserialize(const string &data) {
        vector<string> strings;
        for (int i = 1; i < data.size(); ++i) {
            if (data[i] == ']' || data[i] == '[') {
                continue;
            }
            int k = 0;
            while (!(data[i + k] == ',' || data[i + k] == ']')) {
                ++k;
            }
            strings.push_back(data.substr(i, k));
            i += k;
        }
        queue<TreeNode *> q;
        TreeNode *res;
        if (strings.empty()) {
            return nullptr;
        }
        if (strings[0] == "n") {
            return nullptr;
        } else {
            res = new TreeNode(stoi(strings[0]));
            q.push(res);
        }
        for (int i = 1; i < strings.size(); ++i) {
            if (strings[i] == "n") {
                q.front()->left = nullptr;
            } else {
                q.front()->left = new TreeNode(stoi(strings[i]));
                q.push(q.front()->left);
            }
            ++i;
            if (strings[i] == "n") {
                q.front()->right = nullptr;
            } else {
                q.front()->right = new TreeNode(stoi(strings[i]));
                q.push(q.front()->right);
            }
            q.pop();
        }
        return res;
    }
};

// Your Codec object will be instantiated and called as such:
// Codec ser, deser;
// TreeNode* ans = deser.deserialize(ser.serialize(root));