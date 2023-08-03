//
// Created by David Chen on 8/2/23.
//
#include<iostream>
#include <vector>
#include <algorithm>
#include <queue>
#include <map>
#include <array>
#include <set>
#include <stack>
#include <deque>
#include <algorithm>
#include <cmath>
#include <cctype>
#include <sstream>
#include <regex>
#include <unordered_map>
#include <unordered_set>
#include <numeric>
#include <climits>

using namespace std;

struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;

    TreeNode() : val(0), left(nullptr), right(nullptr) {}

    explicit TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}

    TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
};

struct ListNode {
    int val;
    ListNode *next;

    ListNode(int x) : val(x), next(nullptr) {}
};

class Solution {
public:
    vector<string> letterCombinations(const string &digits) {
        if (digits.empty()) {
            return {};
        }
        unordered_map<char, string> chars{{'1', ""},
                                          {'2', "abc"},
                                          {'3', "def"},
                                          {'4', "ghi"},
                                          {'5', "jkl"},
                                          {'6', "mno"},
                                          {'7', "pqrs"},
                                          {'8', "tuv"},
                                          {'9', "wxyz"},
                                          {'0', ""}};
        vector<string> next = {""};
        vector<string> temp;
        for (auto i: digits) {
            for (auto c: chars[i]) {
                for (auto k: next) {
                    k.push_back(c);
                    temp.emplace_back(k);
                }
            }
            next.swap(temp);
            temp.clear();
        }
        return next;
    }

};

int main() {
    return 0;
}