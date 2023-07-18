//
// Created by david on 23-7-18.
//
#include <iostream>
#include <unordered_map>
#include <queue>

class LRUCache {
public:

    int capacity;
    std::unordered_map<int, int> data;
    std::unordered_map<int, int> times;
    std::queue<int> q;

    LRUCache(int capacity) {
        this->capacity = capacity;
        this->data.reserve(capacity);
    }

    int get(int key) {
        if (data.count(key)) {
            q.push(key);
            ++times[key];
            return data[key];
        } else {
            return -1;
        }
    }

    void put(int key, int value) {
        if (data.count(key) || data.size() < capacity) {
            data[key] = value;
            q.push(key);
            ++times[key];
        } else {
            int eraseElement;
            while (true) {
                eraseElement = q.front();
                q.pop();
                if (--times[eraseElement]) {
                    continue;
                } else {
                    break;

                }
            }
            data.erase(eraseElement);
            q.push(key);
            data[key] = value;
            ++times[key];
        }
    }
};

/**
 * Your LRUCache object will be instantiated and called as such:
 * LRUCache* obj = new LRUCache(capacity);
 * int param_1 = obj->get(key);
 * obj->put(key,value);
 */