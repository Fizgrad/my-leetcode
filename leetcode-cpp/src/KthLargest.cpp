//
// Created by David Chen on 5/23/23.
//
#include <iostream>
#include <queue>
#include <vector>

using namespace std;

class KthLargest {
public:
    priority_queue<int, vector<int>, less<int>> pq;
    int k;

    KthLargest(int k, vector<int> &nums) {
        this->k = k;
        for (auto i: nums) {
            pq.push(-i);
            if (pq.size() > k) {
                pq.pop();
            }
        }
    }

    int add(int val) {
        pq.push(-val);
        if (pq.size() > k) {
            pq.pop();
        }
        return -pq.top();
    }
};

int main() { return 0; }