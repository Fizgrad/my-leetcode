//
// Created by David Chen on 3/28/23.
//

#include <iostream>
#include <vector>

using namespace std;

class DutchNationalFlagPartitioning {
public:
    static void swap(int &a, int &b) {
        int temp = a;
        a = b;
        b = temp;
    }

    static void sort(vector<int> &container) {
        auto lo = container.begin();
        auto hi = container.end() - 1;
        sort(lo, hi);
    }

    static void sort(vector<int>::iterator low, vector<int>::iterator high) {
        if (high - low <= 0) {
            return;
        }
        int a = *low;
        auto lo = low;
        auto hi = high;
        auto i = lo + 1;
        while (i <= hi) {
            if (*i < a) {
                swap(*i, *lo);
                ++i, ++lo;
            } else if (*i == a) {
                ++i;
            } else if (*i > a) {
                swap(*i, *hi);
                --hi;
            }
        }
        sort(low, lo - 1);
        sort(hi + 1, high);
    }
};

int main() {
    vector<int> a{3, 234, 12, 312, 234, 324, 12, 412, 4, 2, 2, 2, 2, 2, 2, 4124, 5, 234, 124, 123};
    DutchNationalFlagPartitioning::sort(a);
    for (auto i: a) {
        cout << i << " ";
    }
}
