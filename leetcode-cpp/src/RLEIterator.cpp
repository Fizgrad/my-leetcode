//
// Created by David Chen on 4/13/23.
//

#include <iostream>
#include <vector>

using namespace std;

class RLEIterator {
public:
    int index = 0;
    vector<int> &encoding;

    RLEIterator(vector<int> &encoding) : encoding(encoding) {}

    int next(int n) {
        while (n > 0) {
            if (index >= encoding.size()) {
                return -1;
            }
            if (encoding[index] <= 0) {
                index += 2;
            }
            int pace = min(encoding[index], n);
            encoding[index] -= pace;
            n -= pace;
        }
        return encoding[index + 1];
    }
};

int main() { return 0; }
/**
 * Your RLEIterator object will be instantiated and called as such:
 * RLEIterator* obj = new RLEIterator(encoding);
 * int param_1 = obj->next(n);
 */