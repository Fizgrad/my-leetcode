//
// Created by David Chen on 5/30/23.
//

#include <vector>

struct Node {
    Node *next = nullptr;
    int val = 0;
};

using namespace std;

class MyHashSet {
public:
    vector<Node *> arr;
    static constexpr int mod = 1007;

    MyHashSet() : arr(1007, nullptr) {
    }

    void add(int key) {
        Node *pt = arr[key % mod];
        if (pt == nullptr) {
            arr[key % mod] = new Node();
            arr[key % mod]->val = key;
            return;
        }
        if (pt->val == key) {
            return;
        }
        while (pt->next != nullptr) {
            if (pt->next->val == key) {
                return;
            }
            pt = pt->next;
        }
        pt->next = new Node();
        pt->next->val = key;
    }

    void remove(int key) {
        Node *pt = arr[key % mod];
        if (pt == nullptr) {
            return;
        }
        if (pt->val == key) {
            arr[key % mod] = pt->next;
            delete pt;
            return;
        }
        while (pt->next != nullptr) {
            if (pt->next->val == key) {
                auto temp = pt->next->next;
                delete pt->next;
                pt->next = temp;
                return;
            }
            pt = pt->next;
        }
    }

    bool contains(int key) {
        Node *pt = arr[key % mod];
        if (pt == nullptr) {
            return false;
        }
        while (pt != nullptr) {
            if (pt->val == key) {
                return true;
            }
            pt = pt->next;
        }
        return false;
    }
};

int main() { return 0; }
/**
 * Your MyHashSet object will be instantiated and called as such:
 * MyHashSet* obj = new MyHashSet();
 * obj->add(key);
 * obj->remove(key);
 * bool param_3 = obj->contains(key);
 */