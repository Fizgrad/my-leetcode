//
// Created by David Chen on 4/8/23.
//
#include <iostream>
#include <queue>
#include <unordered_map>
#include <vector>

using namespace std;

// Definition for a Node.
class Node {
public:
    int val;
    vector<Node *> neighbors;

    Node() {
        val = 0;
        neighbors = vector<Node *>();
    }

    Node(int _val) {
        val = _val;
        neighbors = vector<Node *>();
    }

    Node(int _val, vector<Node *> _neighbors) {
        val = _val;
        neighbors = _neighbors;
    }
};


class Solution {
public:
    Node *cloneGraph(Node *node) {
        unordered_map<int, Node *> umap;
        queue<Node *> q;
        q.push(node);
        if (node == nullptr) {
            return nullptr;
        }
        auto res = (umap[node->val] = new Node(node->val));
        do {
            auto front = q.front();
            q.pop();
            auto new_front = umap[front->val];
            for (auto next: front->neighbors) {
                bool flag;
                Node *new_next;
                auto i = umap.end();
                if ((flag = ((i = umap.find(next->val)) == umap.end()))) {
                    umap[next->val] = (new_next = new Node(next->val));
                } else {
                    new_next = i->second;
                }
                new_front->neighbors.push_back(new_next);
                if (flag) {
                    q.push(next);
                }
            }
        } while (!q.empty());
        return res;
    }
};

int main() { return 0; }