//
// Created by David on 2023/8/28.
//
#include <iostream>
#include <queue>

using namespace std;


class MyStack {
public:
    queue<int> q1;
    queue<int> q_temp;

    MyStack() {
    }

    void push(int x) {
        q1.push(x);
    }

    int pop() {
        q_temp = queue<int>();
        while (q1.size() > 1) {
            q_temp.push(q1.front());
            q1.pop();
        }
        int res = q1.front();
        q1.pop();
        while (!q_temp.empty()) {
            q1.push(q_temp.front());
            q_temp.pop();
        }
        return res;
    }

    int top() {
        q_temp = queue<int>();
        while (q1.size() > 1) {
            q_temp.push(q1.front());
            q1.pop();
        }
        int res = q1.front();
        q_temp.push(q1.front());
        q1.pop();
        while (!q_temp.empty()) {
            q1.push(q_temp.front());
            q_temp.pop();
        }
        return res;
    }

    bool empty() {
        return q1.empty();
    }
};

int main() { return 0; }
/**
 * Your MyStack object will be instantiated and called as such:
 * MyStack* obj = new MyStack();
 * obj->push(x);
 * int param_2 = obj->pop();
 * int param_3 = obj->top();
 * bool param_4 = obj->empty();
 */