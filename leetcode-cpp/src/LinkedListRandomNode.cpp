//
// Created by David Chen on 3/10/23.
//
/**
 * Definition for singly-linked list.
 */
#include <iostream>
#include <vector>
#include <random>
using namespace  std;


struct ListNode {
    int val;

    ListNode *next;

    ListNode() : val(0), next(nullptr) {}

    ListNode(int x) : val(x), next(nullptr) {}

    ListNode(int x, ListNode *next) : val(x), next(next) {}
};

class Solution {
public:
    vector<int> vals;
    default_random_engine e;

    Solution(ListNode *head) {

        while(head!= nullptr){
            this->vals.push_back(head->val);
            head = head->next;
        }
    }

    int getRandom() {
        int n = vals.size();
        uniform_int_distribution<unsigned> u(0,n);
        return vals[u(e)];
    }
};

/**
 * Your Solution object will be instantiated and called as such:
 * Solution* obj = new Solution(head);
 * int param_1 = obj->getRandom();
 */