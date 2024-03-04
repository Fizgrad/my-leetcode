//
// Created by David Chen on 3/19/23.
//
#include <algorithm>
#include <iostream>
#include <unordered_map>

using namespace std;

struct TrieNode {
    unordered_map<char, TrieNode *> map;
    bool isEnd = false;
};

class WordDictionary {
public:
    TrieNode *head;

    WordDictionary() : head(new TrieNode()) {}

    void addWord(const string &word) {
        auto temp = head;
        for (auto i: word) {
            auto iter = temp->map.find(i);
            if (iter == temp->map.end()) {
                temp = (temp->map[i] = new TrieNode());
            } else {
                temp = iter->second;
            }
        }
        temp->isEnd = true;
    }

    bool static searchDFS(TrieNode *temp, const string &word, string::size_type pos) {
        if (pos >= word.size()) {
            return temp->isEnd;
        } else if (word[pos] == '.') {
            auto f = [&](auto input) {
                return searchDFS(input.second, word, pos + 1);
            };
            return std::any_of(temp->map.begin(), temp->map.end(), f);
        } else {
            auto iter = temp->map.find(word[pos]);
            if (iter == temp->map.end()) {
                return false;
            } else {
                return searchDFS(iter->second, word, pos + 1);
            }
        }
    }

    bool search(const string &word) {
        return searchDFS(head, word, 0);
    }
};

int main() { return 0; }
/**
 * Your WordDictionary object will be instantiated and called as such:
 * WordDictionary* obj = new WordDictionary();
 * obj->addWord(word);
 * bool param_2 = obj->search(word);
 */