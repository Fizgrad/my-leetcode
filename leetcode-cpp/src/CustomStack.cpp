#include <vector>

using namespace std;

class CustomStack {
public:
    vector<int> storage;
    int capacity;

    CustomStack(int maxSize) {
        storage.reserve(maxSize);
        capacity = maxSize;
    }

    void push(int x) {
        if (storage.size() == capacity) {
            return;
        }
        storage.push_back(x);
    }

    int pop() {
        if (storage.empty()) {
            return -1;
        }
        int res = storage.back();
        storage.pop_back();
        return res;
    }

    void increment(int k, int val) {
        for (int i = 0; i < storage.size() && i < k; ++i) {
            storage[i] += val;
        }
    }
};

/**
 * Your CustomStack object will be instantiated and called as such:
 * CustomStack* obj = new CustomStack(maxSize);
 * obj->push(x);
 * int param_2 = obj->pop();
 * obj->increment(k,val);
 */

int main() {
    return 0;
}