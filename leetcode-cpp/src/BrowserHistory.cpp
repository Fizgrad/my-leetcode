//
// Created by David Chen on 3/18/23.
//
#include <iostream>
#include <utility>
#include <vector>

using namespace std;

class BrowserHistory {
public:
    vector<string> urls;
    vector<string>::size_type cur = 0;

    explicit BrowserHistory(string homepage) : urls({std::move(homepage)}) {}

    void visit(string url) {
        cur += 1;
        urls.resize(cur + 1);
        urls[cur] = std::move(url);
    }

    string back(unsigned int steps) {
        // cannot write cur -steps >= 0, because cur and steps are unsigned which means cur - signed always > 0
        if (cur >= steps) {
            cur = cur - steps;
        } else {
            cur = 0;
        }
        return urls[cur];
    }

    string forward(unsigned int steps) {
        if (cur + steps >= urls.size()) {
            cur = urls.size() - 1;
        } else {
            cur = cur + steps;
        }
        return urls[cur];
    }
};

/**
 * Your BrowserHistory object will be instantiated and called as such:
 * BrowserHistory* obj = new BrowserHistory(homepage);
 * obj->visit(url);
 * string param_2 = obj->back(steps);
 * string param_3 = obj->forward(steps);
 */