
#include <algorithm>
#include <array>
#include <bitset>
#include <cctype>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <deque>
#include <functional>
#include <ios>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <ostream>
#include <queue>
#include <random>
#include <regex>
#include <set>
#include <sstream>
#include <stack>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <valarray>
#include <vector>

using namespace std;

class Solution {
public:
    int numOfUnplacedFruits(vector<int> &fruits, vector<int> &baskets) {

        vector<int> st;

        auto getMid = [](int s, int e) {
            return s + (e - s) / 2;
        };

        auto MinUtil = [&](auto &&MinUtil, int ss, int se, int l, int r, int node) -> int {
            if (l > r) return 1e9;// 返回一个大数，避免-1干扰min
            if (l <= ss && r >= se) return st[node];
            if (se < l || ss > r) return 1e9;// 返回一个大数
            int mid = getMid(ss, se);
            return min(MinUtil(MinUtil, ss, mid, l, r, 2 * node + 1),
                       MinUtil(MinUtil, mid + 1, se, l, r, 2 * node + 2));
        };

        auto updateValue = [&](auto &&updateValue, int ss, int se, int index, int value, int node) -> void {
            if (ss == se) {
                st[node] = value;
            } else {
                int mid = getMid(ss, se);
                if (index >= ss && index <= mid)
                    updateValue(updateValue, ss, mid, index, value, 2 * node + 1);
                else
                    updateValue(updateValue, mid + 1, se, index, value, 2 * node + 2);
                st[node] = min(st[2 * node + 1], st[2 * node + 2]);
            }
        };

        auto getMin = [&](int n, int l, int r) {
            return MinUtil(MinUtil, 0, n - 1, l, r, 0);
        };

        auto constructSTUtil = [&](auto &&constructSTUtil, vector<int> &arr, int ss, int se, int si) -> int {
            if (ss == se) {
                st[si] = arr[ss];
                return arr[ss];
            }
            int mid = getMid(ss, se);
            st[si] = min(constructSTUtil(constructSTUtil, arr, ss, mid, si * 2 + 1),
                         constructSTUtil(constructSTUtil, arr, mid + 1, se, si * 2 + 2));
            return st[si];
        };

        auto constructST = [&](vector<int> &arr) {
            int n = arr.size();
            if (n == 0) return;
            int x = (int) (ceil(log2(n)));
            int max_size = 2 * (int) pow(2, x) - 1;
            st.assign(max_size, 1e9);// 初始化为一个大数
            constructSTUtil(constructSTUtil, arr, 0, n - 1, 0);
        };


        int place = 0;
        int n = baskets.size();
        if (n == 0) return fruits.size();

        vector<int> sortBasketsIndices(n);
        std::iota(sortBasketsIndices.begin(), sortBasketsIndices.end(), 0);

        auto cmp = [&](int a, int b) {
            return baskets[a] < baskets[b];
        };
        std::sort(sortBasketsIndices.begin(), sortBasketsIndices.end(), cmp);

        vector<int> posInSorted(n);
        for (int i = 0; i < n; ++i) {
            posInSorted[sortBasketsIndices[i]] = i;
        }

        constructST(sortBasketsIndices);

        for (int fruit_size: fruits) {
            int left = 0;
            int right = n - 1;
            int chosen_pos = -1;

            while (left <= right) {
                int mid = left + (right - left) / 2;
                if (baskets[sortBasketsIndices[mid]] < fruit_size) {
                    left = mid + 1;
                } else {
                    chosen_pos = mid;
                    right = mid - 1;
                }
            }

            if (chosen_pos != -1) {

                int original_basket_index = getMin(n, chosen_pos, n - 1);

                if (original_basket_index < 1e9) {
                    ++place;
                    int sortedPosToUpdate = posInSorted[original_basket_index];
                    updateValue(updateValue, 0, n - 1, sortedPosToUpdate, 1e9, 0);
                }
            }
        }
        return fruits.size() - place;
    }

    int maxCollectedFruits(vector<vector<int>> &fruits) {
        int n = fruits.size();
        int res = 0;
        for (int i = 0; i < n; ++i) {
            res += fruits[i][i];
            fruits[i][i] = 0;
        }
        vector<vector<int>> dp(n, vector<int>(n, -1));
        dp[0][n - 1] = fruits[0][n - 1];
        for (int i = 1; i < n; ++i) {
            for (int j = i; j < n; ++j) {
                for (int k = -1; k <= 1; ++k) {
                    if (j + k >= n || j + k < 0 || dp[i - 1][j + k] == -1) continue;
                    dp[i][j] = max(dp[i][j], fruits[i][j] + dp[i - 1][j + k]);
                }
            }
        }
        res += dp[n - 1][n - 1];
        for (int i = 0; i < n; ++i) {
            dp[i][i] = 0;
        }
        dp[n - 1][0] = fruits[n - 1][0];
        for (int i = 1; i < n; ++i) {
            for (int j = i; j < n; ++j) {
                for (int k = -1; k <= 1; ++k) {
                    if (j + k >= n || j + k < 0 || dp[j + k][i - 1] == -1) continue;
                    dp[j][i] = max(dp[j][i], fruits[j][i] + dp[j + k][i - 1]);
                }
            }
        }
        res += dp[n - 1][n - 1];
        return res;
    }

    bool isPowerOfTwo(int n) {
        return n != -2147483648 && __builtin_popcount(n) == 1;
    }

    bool reorderedPowerOf2(int n) {
        set<long long int> sets;
        long long int powerOfTen[10] = {1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000000};
        for (long long int i = 1; i < std::numeric_limits<int>::max(); i <<= 1) {
            long long int tmp = i;
            long long int res = 0;
            while (tmp >= 1) {
                res += powerOfTen[tmp % 10];
                tmp /= 10;
            }
            sets.insert(res);
        }
        long long int tmp = n;
        long long int res = 0;
        while (tmp >= 1) {
            res += powerOfTen[tmp % 10];
            tmp /= 10;
        }
        return sets.contains(res);
    }

    vector<int> productQueries(int n, vector<vector<int>> &queries) {
        const int MOD = 1e9 + 7;
        vector<int> powers;
        for (int i = 0; (1ll << i) <= n; ++i) {
            if ((1ll << i) & n)
                powers.emplace_back(i);
        }
        vector<int> prefix(powers.size() + 1, 0);
        for (int i = 0; i < powers.size(); ++i) {
            prefix[i + 1] = prefix[i] + powers[i];
        }
        vector<int> res(queries.size());
        auto fast_pow = [&](long long a, long long b) -> long long {
            long long res = 1;
            while (b >= 1) {
                if (b % 2 == 1) {
                    res = res * a % MOD;
                }
                b >>= 1;
                a = a * a % MOD;
            }
            return res;
        };
        for (int i = 0; i < queries.size(); ++i) {
            res[i] = fast_pow(2, prefix[queries[i][1] + 1] - prefix[queries[i][0]]);
        }
        return res;
    }
};

int main() {
    [&out = std::cout]() -> void {
        out << "Hello World\n"
            << std::endl;
    }();
    return 0;
}