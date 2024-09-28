#include <iostream>
#include <vector>

using namespace std;
class MyCircularDeque {
public:
    struct Node {
        Node *prev = nullptr;
        Node *next = nullptr;
        int val = 0;
    };

    vector<Node> storage;
    Node *front;
    Node *rear;
    int size = 0;
    int capacity = 0;

    MyCircularDeque(int k) : storage(k), front(nullptr), rear(nullptr) {
        capacity = k;
        for (auto i = 0; i < k - 1; ++i) {
            storage[i].next = &storage[i + 1];
        }
        storage[k - 1].next = &storage[0];
        for (auto i = 1; i < k; ++i) {
            storage[i].prev = &storage[i - 1];
        }
        storage[0].prev = &storage[k - 1];
        front = &storage[0];
        rear = &storage[k - 1];
    }

    bool insertFront(int value) {
        if (isFull()) {
            return false;
        }
        front->prev->val = value;
        front = front->prev;
        ++size;
        return true;
    }

    bool insertLast(int value) {
        if (isFull()) {
            return false;
        }
        ++size;
        rear->next->val = value;
        rear = rear->next;
        return true;
    }

    bool deleteFront() {
        if (isEmpty()) {
            return false;
        }
        --size;
        front = front->next;
        return true;
    }

    bool deleteLast() {
        if (isEmpty()) {
            return false;
        }
        --size;
        rear = rear->prev;
        return true;
    }

    int getFront() {
        if (size == 0) {
            return -1;
        }
        return front->val;
    }

    int getRear() {
        if (size == 0) {
            return -1;
        }
        return rear->val;
    }

    bool isEmpty() {
        return size == 0;
    }

    bool isFull() {
        return size == capacity;
    }

    void display() {
        for (int i = 0; i < capacity; ++i) {
            std::cout << storage[i].val << " ";
        }
        std::cout << std::endl;
    }
};

int main() {
    return 0;
}

/**
 * Your MyCircularDeque object will be instantiated and called as such:
 * MyCircularDeque* obj = new MyCircularDeque(k);
 * bool param_1 = obj->insertFront(value);
 * bool param_2 = obj->insertLast(value);
 * bool param_3 = obj->deleteFront();
 * bool param_4 = obj->deleteLast();
 * int param_5 = obj->getFront();
 * int param_6 = obj->getRear();
 * bool param_7 = obj->isEmpty();
 * bool param_8 = obj->isFull();
 */