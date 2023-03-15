//
// Created by David Chen on 3/15/23.
//

template<class T>
struct Node {
    Node<T> *next;
    T val;

    Node() = default;

    Node(T val, Node<T> *next) : val(val), next(next) {}
};

class MyLinkedList {
private:
    int size;
    Node<int> *head;

public:
    MyLinkedList() : size(0), head(new Node<int>(-1, nullptr)) {}

    int get(int index) {
        if (size <= index) {
            return -1;
        }
        Node<int> *cur = head->next;
        while (index--) {
            cur = cur->next;
        }
        return cur->val;
    }

    void addAtHead(int val) {
        head->next = new Node<int>(val, head->next);
        ++size;
    }

    void addAtTail(int val) {
        Node<int> *cur = head;
        while (cur->next != nullptr) {
            cur = cur->next;
        }
        ++size;
        cur->next = new Node<int>(val, nullptr);
    }

    void addAtIndex(int index, int val) {
        if (size < index) {
            return;
        }
        Node<int> *cur = head;
        while (index--) {
            cur = cur->next;
        }
        ++size;
        cur->next = new Node<int>(val, cur->next);
    }

    void deleteAtIndex(int index) {
        if (size <= index) {
            return;
        }
        Node<int> *cur = head;
        while (index--) {
            cur = cur->next;
        }
        auto temp = cur->next;
        cur->next = cur->next->next;
        delete temp;
        --size;
    }
};

/**
 * Your MyLinkedList object will be instantiated and called as such:
 * MyLinkedList* obj = new MyLinkedList();
 * int param_1 = obj->get(index);
 * obj->addAtHead(val);
 * obj->addAtTail(val);
 * obj->addAtIndex(index,val);
 * obj->deleteAtIndex(index);
 */