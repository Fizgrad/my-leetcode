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

    int minimumPushes(const string &word) {
        vector<int> count(26, 0);
        for (char c: word) {
            count[c - 'a']++;
        }
        sort(count.begin(), count.end(), greater<int>());
        int pushes = 0;
        for (int i = 0; i < 26; ++i) {
            pushes += count[i] * (i / 8 + 1);
        }
        return pushes;
    }

    vector<int> remainingMethods(int n, int k, vector<vector<int>> &invocations) {
        vector<int> parents(n + 1, 0);
        iota(parents.begin(), parents.end(), 0);
        vector<int> sizes(n + 1, 1);
        auto uf_find = [&](auto &&uf_find, int x) {
            int parent = parents[x + 1];
            int tmp = x + 1;
            while (parent != tmp) {
                tmp = parent;
                parent = parents[parent];
            }
            return parent - 1;
        };
        auto uf_union = [&](int x, int y) {
            int rootX = uf_find(uf_find, x) + 1;
            int rootY = uf_find(uf_find, y) + 1;
            if (rootX != rootY) {
                if (sizes[rootX] < sizes[rootY]) {
                    swap(rootX, rootY);
                }
                parents[rootY] = rootX;
                sizes[rootX] += sizes[rootY];
            }
        };
        vector<vector<int>> graph(n);
        for (auto &inv: invocations) {
            graph[inv[0]].push_back(inv[1]);
            uf_union(inv[0], inv[1]);
        }
        unordered_set<int> suspicious;
        vector<bool> visited(n, false);
        auto dfs = [&](auto &&dfs, int node) {
            if (visited[node]) return;
            visited[node] = true;
            suspicious.emplace(node);
            for (int neighbor: graph[node]) {
                dfs(dfs, neighbor);
            }
        };
        dfs(dfs, k);
        bool removed = true;
        int rootK = uf_find(uf_find, k);
        for (int i = 0; i < n; ++i) {
            if (uf_find(uf_find, i) == rootK && suspicious.find(i) == suspicious.end()) {
                removed = false;
                break;
            }
        }
        vector<int> result;
        if (removed) {
            for (int i = 0; i < n; ++i) {
                if (uf_find(uf_find, i) == rootK) {
                    continue;
                } else {
                    result.push_back(i);
                }
            }
        } else {
            for (int i = 0; i < n; ++i) {
                result.push_back(i);
            }
        }
        return result;
    }

    int smallestNumber(int n, int t) {
        while (n) {
            int sum = 1;
            int temp = n;
            while (temp > 0) {
                sum *= (temp % 10);
                temp /= 10;
            }
            if (sum % t == 0) {
                return n;
            }
            ++n;
        }
        return -1;
    }
};

int main() {
    string s = "ADOBECODEBANC";
    string t = "ABC";
    Solution solution;
    string result = solution.minWindow(s, t);
    cout << "Result: " << result << endl;
    {
        int x = -1;
        unsigned int y = 1;
        cout << x << " " << (x < y ? "<" : ">") << " " << y << endl;// Output: -1 > 1
        cout << "x:" << sizeof(x) << " y: " << sizeof(y) << endl;
        long a = -1;
        unsigned int b = 1;
        cout << a << " " << (a < b ? "<" : ">") << " " << b << endl;// Output: -1 < 1
        cout << "a:" << sizeof(a) << " b: " << sizeof(b) << endl;
    }
    return 0;
}