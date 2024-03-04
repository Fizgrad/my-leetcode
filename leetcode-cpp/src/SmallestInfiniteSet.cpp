//
// Created by David Chen on 4/25/23.
//
#include <iostream>
#include <queue>

using namespace std;

class SmallestInfiniteSet {
public:
    vector<bool> has;
    int small;

    SmallestInfiniteSet() : has(1001, true), small(1) {
    }

    int popSmallest() {
        has[small] = false;
        int res = small++;
        while (!has[small]) {
            ++small;
        }
        return res;
    }

    void addBack(int num) {
        if (!has[num]) {
            has[num] = true;
            if (num < small) {
                small = num;
            }
        }
    }
};

int main() { return 0; }

/**
 * Your SmallestInfiniteSet object will be instantiated and called as such:
 * SmallestInfiniteSet* obj = new SmallestInfiniteSet();
 * int param_1 = obj->popSmallest();
 * obj->addBack(num);
 */