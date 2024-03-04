//
// Created by David Chen on 3/25/23.
//
#include <iostream>
#include <vector>

using namespace std;
struct TrieNode {
    TrieNode *children[26] = {};
    bool isEnd = false;
};


class StreamChecker {
public:
    TrieNode *head;
    string buffer = "";

    StreamChecker(vector<string> &words) : head(new TrieNode()) {
        for (auto &i: words) {
            auto temp = head;
            for (int c = i.size() - 1; c >= 0; --c) {
                auto index = i[c] - 'a';
                if (!temp->children[index]) {
                    temp->children[index] = new TrieNode();
                }
                temp = temp->children[index];
            }
            temp->isEnd = true;
        }
    }

    bool query(char letter) {
        buffer += letter;
        TrieNode *node = head;
        for (int i = buffer.length() - 1; i >= 0; i--) {
            int index = buffer[i] - 'a';
            if (!node->children[index]) {
                return false;
            }
            node = node->children[index];
            if (node->isEnd) {
                return true;
            }
        }
        return false;
    }
};

int main() { return 0; }
/**
 * Your StreamChecker object will be instantiated and called as such:
 * StreamChecker* obj = new StreamChecker(words);
 * bool param_1 = obj->query(letter);
 */