//
// Created by David Chen on 3/18/23.
//
#include<iostream>
#include<unordered_map>

using namespace std;
struct TrieNode {
    unordered_map<char, TrieNode *> children;
    bool isEnd = false;
};

class Trie {
public:
    TrieNode *head;

    Trie() {
        head = new TrieNode();
    }

    void insert(const string &word) {
        auto temp = head;
        for (auto i: word) {
            if (temp->children[i] == nullptr) {
                temp->children[i] = new TrieNode();
            }
            temp = temp->children[i];
        }
        temp->isEnd = true;
    }

    bool search(const string &word) {
        auto temp = head;
        for (auto i: word) {
            temp = temp->children[i];
            if (temp == nullptr) {
                return false;
            }
        }
        return temp->isEnd;
    }

    bool startsWith(const string &prefix) {
        auto temp = head;
        for (auto i: prefix) {
            temp = temp->children[i];
            if (temp == nullptr) {
                return false;
            }
        }
        return temp != nullptr;
    }
};

/**
 * Your Trie object will be instantiated and called as such:
 * Trie* obj = new Trie();
 * obj->insert(word);
 * bool param_2 = obj->search(word);
 * bool param_3 = obj->startsWith(prefix);
 */