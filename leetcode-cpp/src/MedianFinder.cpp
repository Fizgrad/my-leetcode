#include <iostream>
#include <queue>

using namespace std;

class MedianFinder {
public:
    priority_queue<int> decreasing;
    priority_queue<int, vector<int>, greater<int>> increasing;

    MedianFinder() {
    }

    void addNum(int num) {
        if (decreasing.size() == increasing.size()) {
            decreasing.push(num);
            increasing.push(decreasing.top());
            decreasing.pop();
        } else {
            increasing.push(num);
            decreasing.push(increasing.top());
            increasing.pop();
        }

    }

    double findMedian() {
        if (decreasing.size() != increasing.size()) return increasing.top();
        return (decreasing.top() + increasing.top()) / 2.0;
    }
};