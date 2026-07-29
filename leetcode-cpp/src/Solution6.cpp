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
            pop(s[left]);// Note: here is not pop(left) but pop(s[left])
        }
        return start == INT_MAX ? "" : s.substr(start, len);
    }

    string smallestPalindrome(const string &s, int k) {
        int n = s.size();
        vector<int> count(26, 0);
        char mid_char = 0;
        for (char c: s) {
            count[c - 'a']++;
        }
        for (auto i = 0; i < 26; ++i) {
            if (count[i] % 2 == 1) {
                mid_char = 'a' + i;
            }
            count[i] = count[i] / 2;
        }
        auto nCr = [](long long n, long long r) -> long long {
            if (r < 0 || r > n) return 0;
            if (r == 0 || r == n) return 1;
            r = min(r, n - r);
            long long res = 1;
            for (long long i = 1; i <= r; ++i) {
                res *= n - r + i;
                res /= i;
                if (res > 1e9) return 1e9;
            }
            return res;
        };

        auto getPermutation = [&](vector<int> &remainCount) -> long long {
            long long res = 1;
            long long len = 0;
            for (int i = 0; i < 26; ++i) {
                len += remainCount[i];
                res *= nCr(len, remainCount[i]);
                if (res > 1e9) return 1e9;
            }
            return res;
        };
        // Initial check: are there enough permutations?
        if (getPermutation(count) < k) {
            return "";
        }
        // Iteratively build the first half
        string half = "";
        int target_len = n / 2;

        for (int step = 0; step < target_len; ++step) {
            for (int i = 0; i < 26; ++i) {
                if (count[i] > 0) {
                    // Temporarily pick this character
                    count[i]--;

                    long long perms = getPermutation(count);

                    if (perms < k) {
                        // Not enough permutations in this branch, skip it
                        k -= perms;
                        count[i]++;// Restore and try next character
                    } else {
                        // The k-th permutation is in this branch. Lock it in!
                        half.push_back('a' + i);
                        break;// Move to the next index in the string
                    }
                }
            }
        }

        // Construct final string
        string result = half;
        if (n % 2 != 0) {
            result.push_back(mid_char);
        }

        // Reverse the first half and append
        reverse(half.begin(), half.end());
        result += half;

        return result;
    }
};

int main() {
    string s = "ADOBECODEBANC";
    string t = "ABC";
    Solution solution;
    string result = solution.minWindow(s, t);
    cout << "Result: " << result << endl;
}