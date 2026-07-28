//
// Created by David Chen
//
#include <algorithm>
#include <array>
#include <bit>
#include <bitset>
#include <cassert>
#include <cctype>
#include <climits>
#include <cmath>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <deque>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <math.h>
#include <new>
#include <numeric>
#include <pthread.h>
#include <queue>
#include <ranges>
#include <ratio>
#include <regex>
#include <set>
#include <sstream>
#include <stack>
#include <string.h>
#include <string>
#include <strings.h>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

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
    string minWindow(const string &s, const string &t) {
        int n = s.size();
        if (t.size() == 1) {
            return s.find(t[0]) == std::string::npos ? "" : t;
        }
        vector<int> lower('z' - 'a' + 1, 0);
        vector<int> upper('Z' - 'A' + 1, 0);
        std::bitset<32> lower_bits(0);
        std::bitset<32> upper_bits(0);
        auto push = [&](char c) {
            if (c >= 'a' && c <= 'z') {
                int index = c - 'a';
                ++lower[index];
                if (lower[index] >= 0) {
                    lower_bits.reset(index);
                }

            } else if (c >= 'A' && c <= 'Z') {
                int index = c - 'A';
                ++upper[index];
                if (upper[index] >= 0) {
                    upper_bits.reset(index);
                }
            }
        };

        auto pop = [&](char c) {
            if (c >= 'a' && c <= 'z') {
                int index = c - 'a';
                --lower[index];
                if (lower[index] < 0) {
                    lower_bits.set(index);
                }

            } else if (c >= 'A' && c <= 'Z') {
                int index = c - 'A';
                --upper[index];
                if (upper[index] < 0) {
                    upper_bits.set(index);
                }
            }
        };

        for (auto i: t) {
            pop(i);
        }

        int right = 0;
        push(s[right]);
        int start = n;
        int len = INT_MAX;
        for (int left = 0; left < n; ++left) {
            while (right + 1 < n && (left == right || lower_bits.any() || upper_bits.any())) {
                push(s[++right]);
            }
            if (!lower_bits.any() && !upper_bits.any()) {
                if (len > right - left + 1) {
                    len = right - left + 1;
                    start = left;
                }
            }
            pop(s[left]); // Note: here is not pop(left) but pop(s[left])
        }
        return start == INT_MAX ? "" : s.substr(start, len);
    }
};

int main() {
    string s = "ADOBECODEBANC";
    string t = "ABC";
    Solution solution;
    string result = solution.minWindow(s, t);
    cout << "Result: " << result << endl;
}