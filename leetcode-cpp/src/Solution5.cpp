
#include <algorithm>
#include <array>
#include <bit>
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

    int numberOfWays(int n, int x) {
        constexpr int mod = 1e9 + 7;
        vector<vector<int>> dp(n + 1, vector<int>(n + 1, 0));
        vector<long long int> powers(n + 1, 0);
        for (int i = 0; i <= n; ++i) {
            long long int num = 1;
            for (int k = 0; k < x; ++k) {
                num *= i;
            }
            powers[i] = num;
        }

        dp[0][0] = 1;
        for (int i = 0; i <= n; ++i) {
            int prefix = 0;
            for (int k = 1; k <= n; ++k) {
                prefix = (prefix + dp[i][k - 1]) % mod;
                long long int num = powers[k];
                if (i + num <= n) {
                    dp[i + num][k] = prefix;
                } else {
                    break;
                }
            }
        }
        return std::accumulate(dp[n].begin(), dp[n].end(), 0ll) % mod;
    }

    bool isPowerOfThree(int n) {
        return n > 0 ? !(1162261467 % n) : false;
    }

    string largestGoodInteger(const string &num) {
        char prev = ' ';
        int time = 0;
        string res = "";
        for (auto c: num) {
            if (c == prev) {
                ++time;
                if (time == 3) {
                    res = max(res, string(3, c));
                }
            } else {
                time = 1;
            }
            prev = c;
        }
        return res;
    }

    bool isPowerOfFour(int n) {
        return n > 0 && __builtin_popcount(n) == 1 && (n - 1) % 3 == 0;
    }

    int maximum69Number(int num) {
        int res = num;
        for (int i = 1; i <= num; i *= 10) {
            if (num / i % 10 == 6) {
                res = max(res, num + 3 * i);
            }
        }
        return res;
    }

    double new21Game(int n, int k, int maxPts) {
        if (k == 0) return 1;
        if (k - 1 + maxPts <= n) return 1;
        if (n < k) return 0;

        vector<double> dp(k + maxPts, 0);
        double prefix = 0;
        double prob = 1.0 / maxPts;
        dp[0] = 1;
        for (int i = 1; i < k + maxPts; ++i) {
            if (i <= k)
                prefix += dp[i - 1];
            if (i - maxPts > 0) {
                prefix -= dp[i - maxPts - 1];
            }
            dp[i] = prefix * prob;
        }
        double res = 0;
        for (int i = k; i <= n; ++i) {
            res += dp[i];
        }
        return res;
    }

    bool judgePoint24(vector<int> &cards) {
        constexpr double delta = 1e-5;
        auto judgePoint24Double = [&](auto &&judgePoint24Double, vector<double> &cards) {
            if (cards.size() == 1) return cards.front() < 24 + delta && cards.front() > 24 - delta;
            for (int i = 0; i < cards.size(); ++i) {
                for (int j = i + 1; j < cards.size(); ++j) {
                    vector<double> newCards;
                    for (int k = 0; k < cards.size(); ++k) {
                        if (k != i && k != j) {
                            newCards.emplace_back(cards[k]);
                        }
                    }
                    double a = cards[i];
                    double b = cards[j];
                    newCards.emplace_back(a - b);
                    if (judgePoint24Double(judgePoint24Double, newCards)) {
                        return true;
                    }
                    newCards.pop_back();
                    newCards.emplace_back(a + b);
                    if (judgePoint24Double(judgePoint24Double, newCards)) {
                        return true;
                    }
                    newCards.pop_back();
                    newCards.emplace_back(b - a);
                    if (judgePoint24Double(judgePoint24Double, newCards)) {
                        return true;
                    }
                    newCards.pop_back();
                    newCards.emplace_back(a * b);
                    if (judgePoint24Double(judgePoint24Double, newCards)) {
                        return true;
                    }
                    newCards.pop_back();
                    if (b != 0) {
                        newCards.emplace_back(a / b);
                        if (judgePoint24Double(judgePoint24Double, newCards)) {
                            return true;
                        }
                        newCards.pop_back();
                    }
                    if (a != 0) {
                        newCards.emplace_back(b / a);
                        if (judgePoint24Double(judgePoint24Double, newCards)) {
                            return true;
                        }
                        newCards.pop_back();
                    }
                }
            }
            return false;
        };
        vector<double> newCards;
        for (auto i: cards) {
            newCards.emplace_back(i);
        }
        return judgePoint24Double(judgePoint24Double, newCards);
    }

    long long zeroFilledSubarray(vector<int> &nums) {
        long long res = 0;
        int prev = 0;
        for (auto i: nums) {
            if (i == 0) {
                res += (++prev);
            } else {
                prev = 0;
            }
        }
        return res;
    }

    int countSquares(vector<vector<int>> &matrix) {
        int res = 0;
        int n = matrix.size();
        int m = matrix.front().size();
        vector<vector<int>> dp(n + 1, vector<int>(m + 1, 0));
        for (int i = 1; i <= n; ++i) {
            for (int j = 1; j <= m; ++j) {
                if (matrix[i - 1][j - 1] == 1) {
                    dp[i][j] = min(dp[i][j - 1], min(dp[i - 1][j], dp[i - 1][j - 1])) + 1;
                    res += dp[i][j];
                }
            }
        }
        return res;
    }

    int numSubmat(vector<vector<int>> &mat) {
        int res = 0;
        int n = mat.size();
        int m = mat.front().size();
        vector<vector<int>> up(n + 1, vector<int>(m + 1, 0));
        for (int i = 1; i <= n; ++i) {
            for (int j = 1; j <= m; ++j) {
                if (mat[i - 1][j - 1] == 1) {
                    up[i][j] = up[i - 1][j] + 1;
                } else {
                    up[i][j] = 0;
                }
            }
        }
        vector<vector<int>> left(n + 1, vector<int>(m + 1, 0));
        for (int i = 1; i <= n; ++i) {
            for (int j = 1; j <= m; ++j) {
                if (mat[i - 1][j - 1] == 1) {
                    left[i][j] = left[i][j - 1] + 1;
                    int minLeft = left[i][j];
                    for (int k = 0; k < up[i][j]; ++k) {
                        int x = i - k;
                        int y = j;
                        minLeft = min(minLeft, left[x][y]);
                        res += minLeft;
                    }
                } else {
                    left[i][j] = 0;
                }
            }
        }
        return res;
    }

    int minimumArea(vector<vector<int>> &grid) {
        int n = grid.size();
        int m = grid.front().size();
        int maxI = 0;
        int minI = n;
        int maxJ = 0;
        int minJ = m;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                if (grid[i][j] == 1) {
                    maxI = max(maxI, i);
                    maxJ = max(j, maxJ);
                    minI = min(minI, i);
                    minJ = min(j, minJ);
                }
            }
        }
        return (maxI - minI + 1) * (maxJ - minJ + 1);
    }

    int minimumSum(vector<vector<int>> &grid) {
        int n = grid.size();
        int m = grid.front().size();
        int ans = m * n;
        auto calMinRect = [&](int x1, int y1, int x2, int y2) {
            if (x1 >= n || x1 < 0 || y1 >= m || y1 < 0 || x2 >= n || x2 < 0 || y2 >= m || y2 < 0) {
                return 0;
            }
            bool flag = false;
            int maxI = x1;
            int minI = x2;
            int maxJ = y1;
            int minJ = y2;
            for (int i = x1; i <= x2; ++i) {
                for (int j = y1; j <= y2; ++j) {
                    if (grid[i][j] == 1) {
                        flag = true;
                        maxI = max(maxI, i);
                        maxJ = max(j, maxJ);
                        minI = min(minI, i);
                        minJ = min(j, minJ);
                    }
                }
            }
            if (flag) {
                return (maxI - minI + 1) * (maxJ - minJ + 1);
            } else {
                return 0;
            }
        };
        auto splitVertically = [&](int x1, int y1, int x2, int y2) {
            int res = (y2 - y1 + 1) * (x2 - x1 + 1);
            for (int i = y1; i < y2; ++i) {
                res = min(res, calMinRect(x1, y1, x2, i) + calMinRect(x1, i + 1, x2, y2));
            }
            return res;
        };
        auto splitHorizontally = [&](int x1, int y1, int x2, int y2) {
            int res = (y2 - y1 + 1) * (x2 - x1 + 1);
            for (int i = x1; i < x2; ++i) {
                res = min(res, calMinRect(x1, y1, i, y2) + calMinRect(i + 1, y1, x2, y2));
            }
            return res;
        };

        for (int i = 0; i < n - 1; ++i) {
            for (int j = i + 1; j < n - 1; ++j) {
                ans = min(ans, calMinRect(0, 0, i, m - 1) + calMinRect(i + 1, 0, j, m - 1) + calMinRect(j + 1, 0, n - 1, m - 1));
            }
        }

        for (int i = 0; i < m - 1; ++i) {
            for (int j = i + 1; j < m - 1; ++j) {
                ans = min(ans, calMinRect(0, 0, n - 1, i) + calMinRect(0, i + 1, n - 1, j) + calMinRect(0, j + 1, n - 1, m - 1));
            }
        }

        for (int i = 0; i < n - 1; ++i) {
            ans = min(ans, calMinRect(0, 0, i, m - 1) + splitVertically(i + 1, 0, n - 1, m - 1));
            ans = min(ans, calMinRect(i + 1, 0, n - 1, m - 1) + splitVertically(0, 0, i, m - 1));
        }

        for (int i = 0; i < m - 1; ++i) {
            ans = min(ans, calMinRect(0, 0, n - 1, i) + splitHorizontally(0, i + 1, n - 1, m - 1));
            ans = min(ans, calMinRect(0, i + 1, n - 1, m - 1) + splitHorizontally(0, 0, n - 1, i));
        }
        return ans;
    }

    int longestSubarray(vector<int> &nums) {
        int n = nums.size();
        int res = 0;
        int prev = 0;
        int len = 0;
        for (int i = 0; i < n; ++i) {
            if (nums[i] == 1) {
                ++len;
                res = max(res, len + prev);
            } else {
                if (len > 0) {
                    prev = len;
                    len = 0;
                } else {
                    prev = 0;
                }
            }
        }
        return res == n ? n - 1 : res;
    }

    vector<int> findDiagonalOrder(vector<vector<int>> &mat) {
        int dx[2] = {-1, 1};
        int dy[2] = {1, -1};
        int x = 0;
        int y = 0;
        int n = mat.size();
        int m = mat.front().size();
        vector<int> res;
        res.reserve(m * n);
        int direction = 0;
        int count = 0;
        if (n == 1) {
            return mat.front();
        }
        if (m == 1) {
            for (int i = 0; i < n; ++i) {
                res.emplace_back(mat[i][0]);
            }
            return res;
        }
        while (count < m * n) {
            res.emplace_back(mat[x][y]);
            x += dx[direction];
            y += dy[direction];
            ++count;
            if (x < 0 && y < m) {
                x = 0;
                direction = 1 - direction;
            } else if (y < 0 && x < n) {
                y = 0;
                direction = 1 - direction;
            } else if (x < 0 && y >= m) {
                x = 1;
                y = m - 1;
                direction = 1 - direction;
            } else if (x >= n && y < 0) {
                y = 1;
                x = n - 1;
                direction = 1 - direction;
            } else if (x >= n && y >= 0 && y < m) {
                y += 2;
                x = n - 1;
                direction = 1 - direction;
            } else if (y >= m && x >= 0 && x < n) {
                x += 2;
                y = m - 1;
                direction = 1 - direction;
            }
        }
        return res;
    }

    int areaOfMaxDiagonal(vector<vector<int>> &dimensions) {
        long long int maxDiagSquare = 0;
        long long int res = 0;
        for (auto &i: dimensions) {
            auto diagSquare = 1ll * i[0] * i[0] + 1ll + i[1] * i[1];
            if (diagSquare > maxDiagSquare) {
                maxDiagSquare = diagSquare;
                res = 1ll * i[0] * i[1];
            } else if (diagSquare == maxDiagSquare) {
                res = max(res, 1ll * i[0] * i[1]);
            }
        }
        return res;
    }

    int lenOfVDiagonal(vector<vector<int>> &grid) {
        int n = grid.size();
        int m = grid.front().size();
        int dx[4] = {1, 1, -1, -1};
        int dy[4] = {1, -1, -1, 1};
        vector<vector<vector<int>>> maxLen(4, vector<vector<int>>(n + 2, vector<int>(m + 2, 0)));
        int res = 0;
        for (int i = 0; i < n; ++i) {
            maxLen[2][i][0] = ((grid[i][0] == 0 || grid[i][0] == 2) ? 1 : 0);
            maxLen[1][i][0] = ((grid[i][0] == 0 || grid[i][0] == 2) ? 1 : 0);
            maxLen[3][i][m - 1] = ((grid[i][m - 1] == 0 || grid[i][m - 1] == 2) ? 1 : 0);
            maxLen[0][i][m - 1] = ((grid[i][m - 1] == 0 || grid[i][m - 1] == 2) ? 1 : 0);
        }
        for (int j = 0; j < m; ++j) {
            maxLen[2][0][j] = ((grid[0][j] == 0 || grid[0][j] == 2) ? 1 : 0);
            maxLen[1][n - 1][j] = ((grid[n - 1][j] == 0 || grid[n - 1][j] == 2) ? 1 : 0);
            maxLen[3][0][j] = ((grid[0][j] == 0 || grid[0][j] == 2) ? 1 : 0);
            maxLen[0][n - 1][j] = ((grid[n - 1][j] == 0 || grid[n - 1][j] == 2) ? 1 : 0);
        }
        for (int i = 1; i < n; ++i) {
            for (int j = 1; j < m; ++j) {
                if (grid[i][j] == 0 && grid[i - 1][j - 1] == 2 || grid[i][j] == 2 && grid[i - 1][j - 1] == 0) {
                    maxLen[2][i][j] = maxLen[2][i - 1][j - 1] + 1;
                } else if (grid[i][j] == 2 || grid[i][j] == 0) {
                    maxLen[2][i][j] = 1;
                }
            }
        }

        for (int i = n - 2; i >= 0; --i) {
            for (int j = 1; j < m; ++j) {
                if (grid[i][j] == 0 && grid[i + 1][j - 1] == 2 || grid[i][j] == 2 && grid[i + 1][j - 1] == 0) {
                    maxLen[1][i][j] = maxLen[1][i + 1][j - 1] + 1;
                } else if (grid[i][j] == 2 || grid[i][j] == 0) {
                    maxLen[1][i][j] = 1;
                }
            }
        }
        for (int i = 1; i < n; ++i) {
            for (int j = m - 2; j >= 0; --j) {
                if (grid[i][j] == 0 && grid[i - 1][j + 1] == 2 || grid[i][j] == 2 && grid[i - 1][j + 1] == 0) {
                    maxLen[3][i][j] = maxLen[3][i - 1][j + 1] + 1;
                } else if (grid[i][j] == 2 || grid[i][j] == 0) {
                    maxLen[3][i][j] = 1;
                }
            }
        }
        for (int i = n - 2; i >= 0; --i) {
            for (int j = m - 2; j >= 0; --j) {
                if (grid[i][j] == 0 && grid[i + 1][j + 1] == 2 || grid[i][j] == 2 && grid[i + 1][j + 1] == 0) {
                    maxLen[0][i][j] = maxLen[0][i + 1][j + 1] + 1;
                } else if (grid[i][j] == 2 || grid[i][j] == 0) {
                    maxLen[0][i][j] = 1;
                }
            }
        }
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                if (grid[i][j] == 1) {
                    res = max(1, res);
                    for (int k = 0; k < 4; ++k) {
                        int x = i + dx[k];
                        int y = j + dy[k];
                        int temp = 0;
                        int last = 1;
                        while (true) {
                            if (x < 0 || x >= n || y < 0 || y >= m) {
                                break;
                            }
                            if (last == 1 && grid[x][y] != 2 || last == 2 && grid[x][y] != 0 || last == 0 && grid[x][y] != 2) {
                                break;
                            }
                            res = max(res, ++temp);
                            res = max(res, temp + maxLen[(k + 1) % 4][x][y]);
                            last = grid[x][y];
                            x = x + dx[k];
                            y = y + dy[k];
                        }
                    }
                }
            }
        }
        return res;
    }

    vector<vector<int>> sortMatrix(vector<vector<int>> &grid) {
        int n = grid.size();
        for (int i = 0; i < n; ++i) {
            int x = i;
            int y = 0;
            while (x >= 0 && x < n && y >= 0 && y < n) {
                int num = grid[x][y];
                for (int p = x, q = y; p >= 0 && p < n && q >= 0 && q < n; ++p, ++q) {
                    if (grid[p][q] > num) {
                        num = grid[p][q];
                        std::swap(grid[x][y], grid[p][q]);
                    }
                }
                ++x;
                ++y;
            }
        }
        for (int i = 1; i < n; ++i) {
            int x = 0;
            int y = i;
            while (x >= 0 && x < n && y >= 0 && y < n) {
                int num = grid[x][y];
                for (int p = x, q = y; p >= 0 && p < n && q >= 0 && q < n; ++p, ++q) {
                    if (grid[p][q] < num) {
                        num = grid[p][q];
                        std::swap(grid[x][y], grid[p][q]);
                    }
                }
                ++x;
                ++y;
            }
        }
        return grid;
    }

    long long flowerGame(long long int n, long long m) {
        if (m & 1 && n & 1) {
            return ((1ll + n) >> 1) * ((m - 1) >> 1) + ((n - 1ll) >> 1) * ((m + 1) >> 1);
        } else if (n & 1) {
            return ((1ll + n) >> 1) * ((m) >> 1) + ((n - 1ll) >> 1) * ((m) >> 1);
        } else if (m & 1) {
            return ((n) >> 1ll) * ((m - 1) >> 1) + ((n) >> 1ll) * ((m + 1) >> 1);
        } else {
            return ((n) >> 1ll) * ((m) >> 1ll) + ((n) >> 1ll) * ((m) >> 1ll);
        }
    }

    bool isValidSudoku(vector<vector<char>> &board) {
        int n = board.size();
        vector<int> count(9, 0);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (isdigit(board[i][j])) {
                    if (++count[board[i][j] - '1'] > 1) {
                        return false;
                    }
                }
            }
            std::fill(count.begin(), count.end(), 0);
        }
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (isdigit(board[j][i])) {
                    if (++count[board[j][i] - '1'] > 1) {
                        return false;
                    }
                }
            }
            std::fill(count.begin(), count.end(), 0);
        }
        for (int i = 0; i < n; i += 3) {
            for (int j = 0; j < n; j += 3) {
                for (int k = 0; k < 3; ++k) {
                    for (int l = 0; l < 3; ++l) {
                        if (isdigit(board[j + k][i + l])) {
                            if (++count[board[j + k][i + l] - '1'] > 1) {
                                return false;
                            }
                        }
                    }
                }
                std::fill(count.begin(), count.end(), 0);
            }
        }
        return true;
    }

    double maxAverageRatio(vector<vector<int>> &classes, int extraStudents) {
        int n = classes.size();
        auto cmp = [](const vector<int> &left, const vector<int> &right) {
            int a = left[0], b = left[1];
            int c = right[0], d = right[1];
            return (b - a) * (d + 1ll) * d < (d - c) * (b + 1ll) * b;
        };
        std::make_heap(classes.begin(), classes.end(), cmp);
        while (extraStudents--) {
            std::pop_heap(classes.begin(), classes.end(), cmp);
            ++classes.back().front();
            ++classes.back().back();
            std::push_heap(classes.begin(), classes.end(), cmp);
        }
        double res = 0;
        for (auto &i: classes) {
            res += static_cast<double>(i[0]) / i[1];
        }
        return res / n;
    }

    int numberOfPairs(vector<vector<int>> &points) {
        int n = points.size();
        auto cmp = [&](auto &a, auto &b) {
            return a.front() < b.front() || (a.front() == b.front() && a.back() > b.back());
        };
        std::sort(points.begin(), points.end(), cmp);
        int res = 0;
        for (int i = 0; i < n; ++i) {
            int closest = -1;
            for (int j = i + 1; j < n; ++j) {
                if (points[j][1] > points[i][1]) {
                    continue;
                } else {
                    if (closest == -1 || closest < points[j][1]) {
                        ++res;
                        closest = points[j][1];
                    } else {
                        continue;
                    }
                }
            }
        }
        return res;
    }

    int findClosest(int x, int y, int z) {
        int xz = std::abs(z - x);
        int yz = std::abs(z - y);
        return xz == yz ? 0 : (xz > yz ? 2 : 1);
    }

    int makeTheIntegerZero(int num1, int num2) {
        int i = 0;
        while (true) {
            long long rest = num1 - 1ll * i * num2;
            if (rest < 0) {
                return -1;
            }
            if (rest == 1) {
                if (i == 1) {
                    return 1;
                }
            } else {
                int oneCount = __builtin_popcountll(rest);
                if (oneCount != 0 && oneCount <= i) {
                    return i;
                }
            }
            ++i;
        }
    }

    long long minOperations(vector<vector<int>> &queries) {
        long long res = 0;
        auto log4 = [](auto x) {
            return (31 - std::countl_zero(x)) / 2;
        };
        vector<long long> expSum4(18, 1);
        for (int i = 1; i < 18; i++) {
            expSum4[i] = expSum4[i - 1] + 3LL * i * (1LL << (2 * (i - 1))) + 1;
        }
        auto expSum = [&](unsigned x) -> long long {
            if (x == 0) return 0;
            int log4x = log4(x);
            int r = x - (1 << (2 * log4x));
            return expSum4[log4x] + r * (log4x + 1LL);
        };
        for (auto &q: queries) {
            int l = q[0] - 1, r = q[1];
            res += (expSum(r) - expSum(l) + 1) / 2;
        }
        return res;
    }

    vector<int> sumZero(int n) {
        vector<int> res(n, 0);
        for (int i = 0; 2 * i + 1 < n; ++i) {
            res[2 * i] = (i + 1);
            res[2 * i + 1] = -(i + 1);
        }
        if (n & 1) {
            res.back() = 0;
        }
        return res;
    }

    vector<int> getNoZeroIntegers(int n) {
        int powerOfTen[6] = {1, 10, 100, 1000, 10000, 100000};
        vector<int> digits;
        int remain = n;
        while (remain > 0) {
            digits.emplace_back(remain % 10);
            remain /= 10;
        }
        bool minus = false;
        int num1 = 0;
        for (int i = 0; i < digits.size(); ++i) {
            if (minus && i + 1 == digits.size() - 1 && digits[digits.size() - 1] == 1) {
                num1 += 9 * powerOfTen[i];
                break;
            }
            if (minus && i == digits.size() - 1 && digits[i] == 1) {
                break;
            }
            if (minus) {
                --digits[i];
                minus = false;
            }
            if (digits[i] == 0 || digits[i] == 1) {
                if (i != digits.size() - 1) {
                    minus = true;
                    num1 += 9 * powerOfTen[i];
                } else {
                    num1 += digits[i] * powerOfTen[i];
                }
            } else {
                num1 += (digits[i] - 1) * powerOfTen[i];
            }
        }
        return {num1, n - num1};
    }

    int peopleAwareOfSecret(int n, int delay, int forget) {
        constexpr int mod = 1e9 + 7;
        vector<int> dp(n + 1, 0);
        dp[1] = 1;
        int prefix = 0;
        for (int i = 2; i <= n; ++i) {
            if (i - forget >= 1) {
                prefix = (prefix - dp[i - forget] + mod) % mod;
            }
            if (i - delay >= 1) {
                prefix = (prefix + dp[i - delay]) % mod;
            }
            dp[i] = prefix;
        }
        int res = 0;
        for (int i = n - forget + 1; i <= n; ++i) {
            res = (res + dp[i]) % mod;
        }
        return res;
    }

    int minimumTeachings(int n, vector<vector<int>> &languages, vector<vector<int>> &friendships) {
        int m = languages.size();
        vector<std::bitset<501>> know(m);
        for (int i = 0; i < m; i++)
            for (int l: languages[i]) know[i][l] = 1;

        std::bitset<501> need = 0;
        for (auto &f: friendships) {
            int a = f[0] - 1, b = f[1] - 1;
            if ((know[a] & know[b]).any()) continue;
            need[a] = need[b] = 1;
        }

        if (need.count() == 0) return 0;

        int ans = INT_MAX;
        for (int lang = 1; lang <= n; lang++) {
            int cnt = 0;
            for (int i = 0; i < m; i++) {
                if (need[i] & !know[i][lang]) cnt++;
            }
            ans = min(ans, cnt);
        }

        return ans;
    }

    string sortVowels(const string &s) {
        const string vowels = "AEIOUaeiou";
        vector<int> vowelsCount(vowels.size(), 0);
        for (int i = 0; i < s.size(); ++i) {
            auto iter = vowels.find(s[i]);
            if (iter != string::npos) {
                vowelsCount[iter]++;
            }
        }
        int index = 0;
        string res = s;
        for (int i = 0; i < s.size(); ++i) {
            auto iter = vowels.find(s[i]);
            if (iter != string::npos) {
                while (index < vowelsCount.size() && vowelsCount[index] == 0) {
                    ++index;
                }
                res[i] = vowels[index];
                --vowelsCount[index];
            }
        }
        return res;
    }

    bool doesAliceWin(const string &s) {
        int n = s.size();
        auto isVowel = [](char c) {
            return c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u';
        };
        int vowelsCount = 0;
        for (int i = 0; i < n; ++i) {
            vowelsCount += (isVowel(s[i]) ? 1 : 0);
        }
        if (vowelsCount == 0) {
            return false;
        }
        return true;
    }

    int maxFreqSum(const string &s) {
        int n = s.size();
        const string vowels = "aeiou";
        vector<int> freq(26, 0);
        for (int i = 0; i < n; ++i) {
            freq[s[i] - 'a']++;
        }
        int maxVowels = 0;
        int maxConsonants = 0;
        int maxFreq = 0;
        for (int i = 0; i < 26; ++i) {
            auto iter = vowels.find('a' + i);
            if (iter != string::npos) {
                maxVowels = max(maxVowels, freq[i]);
            } else {
                maxConsonants = max(maxConsonants, freq[i]);
            }
        }
        return maxVowels + maxConsonants;
    }

    vector<string> spellchecker(vector<string> &wordlist, vector<string> &queries) {
        unordered_set<string> wordSet;
        unordered_map<string, string> lowerMap;
        unordered_map<string, string> vowelMap;
        auto devowel = [](const string &s) {
            string res = s;
            for (auto &c: res) {
                if (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u' ||
                    c == 'A' || c == 'E' || c == 'I' || c == 'O' || c == 'U') {
                    c = '*';
                }
            }
            return res;
        };
        for (const auto &word: wordlist) {
            wordSet.insert(word);
            string lowerWord = word;
            std::transform(lowerWord.begin(), lowerWord.end(), lowerWord.begin(), ::tolower);
            if (lowerMap.find(lowerWord) == lowerMap.end()) {
                lowerMap[lowerWord] = word;
            }
            string devowelWord = devowel(lowerWord);
            if (vowelMap.find(devowelWord) == vowelMap.end()) {
                vowelMap[devowelWord] = word;
            }
        }
        vector<string> res;
        for (const auto &query: queries) {
            if (wordSet.find(query) != wordSet.end()) {
                res.emplace_back(query);
                continue;
            }
            string lowerQuery = query;
            std::transform(lowerQuery.begin(), lowerQuery.end(), lowerQuery.begin(), ::tolower);
            if (lowerMap.find(lowerQuery) != lowerMap.end()) {
                res.emplace_back(lowerMap[lowerQuery]);
                continue;
            }
            string devowelQuery = devowel(lowerQuery);
            if (vowelMap.find(devowelQuery) != vowelMap.end()) {
                res.emplace_back(vowelMap[devowelQuery]);
                continue;
            }
            res.emplace_back("");
        }
        return res;
    }

    int canBeTypedWords(const string &text, const string &brokenLetters) {
        if (brokenLetters.size() == 26) return 0;
        int res = 0;
        auto space = text.find(' ');
        decltype(space) index = 0;
        while (space != std::string::npos) {
            bool flag = true;
            for (auto c: brokenLetters) {
                if (std::find(text.begin() + index, text.begin() + space, c) != text.begin() + space) {
                    flag = false;
                    break;
                }
            }
            if (flag)
                ++res;
            index = space + 1;
            space = text.find(' ', space + 1);
        }
        if (index < text.size()) {
            bool flag = true;
            for (auto c: brokenLetters) {
                if (std::find(text.begin() + index, text.end(), c) != text.end()) {
                    flag = false;
                    break;
                }
            }
            if (flag)
                ++res;
        }
        return res;
    }

    vector<int> replaceNonCoprimes(vector<int> &nums) {
        auto gcd = [](int a, int b) {
            while (b) {
                int temp = a % b;
                a = b;
                b = temp;
            }
            return a;
        };
        auto lcm = [&](int a, int b) {
            return a / gcd(a, b) * b;
        };
        vector<int> res;
        for (auto num: nums) {
            while (!res.empty() && gcd(res.back(), num) != 1) {
                num = lcm(res.back(), num);
                res.pop_back();
            }
            res.push_back(num);
        }
        return res;
    }

    int maxFrequencyElements(vector<int> &nums) {
        int max_freq = 1;
        vector<int> times(101, 0);
        int res = 0;
        for (auto i: nums) {
            ++times[i];
            if (times[i] > max_freq) {
                res = times[i];
                max_freq = times[i];
            } else if (times[i] == max_freq) {
                res += times[i];
            }
        }
        return res;
    }

    int compareVersion(const string &version1, const string &version2) {
        int n1 = version1.size(), n2 = version2.size();
        int x1 = 0, x2 = 0;
        for (int i = 0, j = 0; i < n1 || j < n2; i++, j++) {
            while (i < n1 && version1[i] != '.') {
                x1 = 10 * x1 + (version1[i++] - '0');
            }
            while (j < n2 && version2[j] != '.') {
                x2 = 10 * x2 + (version2[j++] - '0');
            }
            if (x1 < x2) return -1;
            else if (x1 > x2)
                return 1;
            x1 = 0;
            x2 = 0;
        }
        return 0;
    }

    string fractionToDecimal(long long numerator, long long denominator) {
        if (numerator == 0) return "0";
        string res;
        if ((numerator ^ denominator) < 0) {
            res = "-";
            numerator = std::abs(numerator);
            denominator = std::abs(denominator);
        }
        long long gcd = [](long long a, long long b) {
            if (a < b) {
                std::swap(a, b);
            }
            while (b) {
                int temp = a % b;
                a = b;
                b = temp;
            }
            return a;
        }(numerator, denominator);
        numerator /= gcd;
        denominator /= gcd;
        long long integerPart = numerator / denominator;
        long long remain = numerator - integerPart * denominator;
        unordered_map<long long, long long> remainSeen;
        res = res + to_string(integerPart);
        if (remain == 0) return res;
        string decimalPart;
        remainSeen[remain] = 0;
        remain *= 10;
        while (remain != 0) {
            long long integerPart = remain / denominator;
            remain = remain - integerPart * denominator;
            decimalPart.push_back('0' + integerPart);
            if (remainSeen.contains(remain)) {
                return res + "." + decimalPart.substr(0, remainSeen[remain]) + "(" + decimalPart.substr(remainSeen[remain]) + ")";
            }
            remainSeen[remain] = decimalPart.size();
            remain *= 10;
        }
        return res + "." + decimalPart;
    }

    int minimumTotal(vector<vector<int>> &triangle) {
        int n = triangle.size();
        vector<int> dp(n, 0);
        dp[0] = triangle[0][0];
        for (int i = 1; i < n; ++i) {
            for (int j = i; j >= 0; --j) {
                if (j == 0) {
                    dp[j] = dp[j] + triangle[i][j];
                } else if (j == i) {
                    dp[j] = dp[j - 1] + triangle[i][j];
                } else {
                    dp[j] = min(dp[j - 1], dp[j]) + triangle[i][j];
                }
            }
        }
        return *std::min_element(dp.begin(), dp.end());
    }

    int triangleNumber(vector<int> &nums) {
        int n = nums.size();
        std::sort(nums.begin(), nums.end());
        int res = 0;
        for (int i = n - 1; i >= 2; --i) {
            int left = 0;
            int right = i - 1;
            while (left < right) {
                if (nums[left] + nums[right] > nums[i]) {
                    res += (right - left);
                    --right;
                } else {
                    ++left;
                }
            }
        }
        return res;
    }

    double largestTriangleArea(vector<vector<int>> &points) {
        int n = points.size();
        double res = 0.0;
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                double x1 = points[i][0];
                double y1 = points[i][1];
                double x2 = points[j][0];
                double y2 = points[j][1];
                double len = std::sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
                for (int k = j + 1; k < n; ++k) {
                    double x3 = points[k][0];
                    double y3 = points[k][1];
                    double dis = std::abs((y1 - y2) * x3 + (x2 - x1) * y3 + x1 * y2 - y1 * x2) / len;
                    res = max(res, len * dis / 2);
                }
            }
        }
        return res;
    }

    int largestPerimeter(vector<int> &nums) {
        std::sort(nums.begin(), nums.end());
        int n = nums.size();
        if (n <= 2) return 0;
        for (int i = n - 1; i >= 2; --i) {
            if (nums[i - 1] + nums[i - 2] > nums[i]) {
                return nums[i - 1] + nums[i - 2] + nums[i];
            }
        }
        return 0;
    }

    int minScoreTriangulation(vector<int> &values) {
        int n = values.size();
        vector<vector<int>> dp(n, vector<int>(n, 0));
        for (int diff = 2; diff < n; ++diff) {
            for (int ind = 0; ind < n - diff; ++ind) {
                int s = ind, e = ind + diff;
                dp[s][e] = std::numeric_limits<int>::max();
                for (int t = s + 1; t < e; ++t) {
                    dp[s][e] = min(dp[s][e], dp[s][t] + dp[t][e] + values[s] * values[t] * values[e]);
                }
            }
        }
        return dp[0][n - 1];
    }

    int maxArea(vector<int> &height) {
        int n = height.size();
        int res = 0;
        int left = 0;
        int right = n - 1;
        int curHeight = min(height[left], height[right]);
        res = max(res, curHeight * (right - left));
        while (left < right) {
            if (height[left] < height[right]) {
                ++left;
            } else {
                --right;
            }
            int curHeight = min(height[left], height[right]);
            res = max(res, curHeight * (right - left));
        }
        return res;
    }

    vector<vector<int>> pacificAtlantic(vector<vector<int>> &heights) {
        int n = heights.size();
        int m = heights.front().size();
        vector<vector<bool>> pacific(n, vector<bool>(m, false));
        vector<vector<bool>> atlantic(n, vector<bool>(m, false));
        vector<pair<int, int>> workSet;
        workSet.reserve(m * n / 2);
        for (int i = 0; i < m; ++i) {
            pacific[0][i] = true;
            atlantic[n - 1][i] = true;
            workSet.emplace_back(0, i);
        }
        for (int i = 0; i < n; ++i) {
            pacific[i][0] = true;
            atlantic[i][m - 1] = true;
            workSet.emplace_back(i, 0);
        }
        constexpr int d[5] = {0, 1, 0, -1, 0};
        auto propagation = [&](auto &visiited) {
            while (workSet.size()) {
                auto [x, y] = workSet.back();
                workSet.pop_back();
                for (int i = 0; i < 4; ++i) {
                    int xx = x + d[i];
                    int yy = y + d[i + 1];
                    if (xx < 0 || yy < 0 || xx >= n || yy >= m) {
                        continue;
                    }
                    if (visiited[xx][yy]) continue;
                    if (heights[xx][yy] >= heights[x][y]) {
                        workSet.emplace_back(xx, yy);
                        visiited[xx][yy] = true;
                    }
                }
            }
        };
        propagation(pacific);
        for (int i = 0; i < m; ++i) {
            atlantic[n - 1][i] = true;
            workSet.emplace_back(n - 1, i);
        }
        for (int i = 0; i < n; ++i) {
            atlantic[i][m - 1] = true;
            workSet.emplace_back(i, m - 1);
        }
        propagation(atlantic);
        vector<vector<int>> res;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                if (atlantic[i][j] && pacific[i][j]) {
                    res.push_back({i, j});
                }
            }
        }
        return res;
    }

    int swimInWater(vector<vector<int>> &grid) {
        int n = grid.size();
        vector<vector<int>> minReachTime(n, vector<int>(n, -1));
        constexpr int d[5] = {0, 1, 0, -1, 0};
        using Coordinate = pair<int, int>;
        using Term = pair<int, Coordinate>;
        priority_queue<Term, vector<Term>, greater<>> pq;
        auto f = [&](int x, int y, int time) -> void {
            for (int k = 0; k < 4; ++k) {
                int xx = x + d[k];
                int yy = y + d[k + 1];
                if (xx < 0 || yy < 0 || xx >= n || yy >= n) {
                    continue;
                }
                if (minReachTime[xx][yy] == -1) {
                    minReachTime[xx][yy] = max(time, grid[xx][yy]);
                    pq.emplace(minReachTime[xx][yy], std::make_pair(xx, yy));
                } else {
                    if (max(time, grid[xx][yy]) < minReachTime[xx][yy]) {
                        minReachTime[xx][yy] = max(time, grid[xx][yy]);
                        pq.emplace(minReachTime[xx][yy], std::make_pair(xx, yy));
                    }
                }
            }
        };
        minReachTime[0][0] = grid[0][0];
        pq.emplace(minReachTime[0][0], std::make_pair(0, 0));
        while (pq.size()) {
            auto [time, Coordinate] = pq.top();
            pq.pop();
            auto [x, y] = Coordinate;
            f(x, y, time);
        }
        return minReachTime[n - 1][n - 1];
    }

    vector<int> avoidFlood(vector<int> &rains) {
        int n = rains.size();
        set<int> dryDays;
        unordered_map<int, int> lakesToDry;
        vector<int> res(n, -1);
        for (int i = 0; i < n; ++i) {
            if (rains[i] == 0) {
                dryDays.emplace(i);
            } else {
                if (lakesToDry.contains(rains[i])) {
                    if (dryDays.empty()) {
                        return {};
                    }
                    auto it = dryDays.upper_bound(lakesToDry[rains[i]]);
                    if (it == dryDays.end()) return {};
                    auto choose = *it;
                    res[choose] = rains[i];
                    dryDays.erase(choose);
                }
                lakesToDry[rains[i]] = i;
            }
        }
        for (auto i: dryDays) {
            res[i] = 1;
        }
        return res;
    }

    vector<int> successfulPairs(vector<int> &spells, vector<int> &potions, long long success) {
        int n = spells.size();
        int m = potions.size();
        std::sort(potions.begin(), potions.end());
        vector<int> res(n, 0);
        for (int i = 0; i < n; ++i) {
            auto it = std::lower_bound(potions.begin(), potions.end(), (success + spells[i] - 1) / spells[i]);
            res[i] = std::distance(it, potions.end());
        }
        return res;
    }

    long long minTime(vector<int> &skill, vector<int> &mana) {
        int n = skill.size();
        int m = mana.size();
        vector<long long> done(n + 1, 0);
        for (int j = 0; j < m; ++j) {
            for (int i = 0; i < n; ++i) {
                done[i + 1] = max(done[i + 1], done[i]) + 1ll * mana[j] * skill[i];
            }
            for (int i = n - 1; i > 0; --i) {
                done[i] = done[i + 1] - 1ll * mana[j] * skill[i];
            }
        }
        return done[n];
    }

    int maximumEnergy(vector<int> &energy, int k) {
        int n = energy.size();
        vector<int> dp(k, 0);
        int res = energy.back();
        for (int i = n - 1; i >= 0; --i) {
            res = max(res, dp[i % k] = dp[i % k] + energy[i]);
        }
        return res;
    }

    long long maximumTotalDamage(vector<int> &power) {
        int n = power.size();
        long long res = 0;
        unordered_map<int, long long> time;
        for (auto i: power) {
            ++time[i];
        }
        std::ranges::sort(power);
        auto iter = std::unique(power.begin(), power.end());
        auto size = std::distance(power.begin(), iter);
        power.resize(size);
        vector<long long> dp(size + 1, 0);
        for (auto i = power.begin(); i != iter; ++i) {
            auto index = std::distance(power.begin(), i);
            long long gain = power[index] * time[power[index]];
            res = max(res, dp[index + 1] = max(dp[index], gain));
            for (int k = index - 1; k >= 0; --k) {
                if (power[index] - power[k] > 2) {
                    res = max(res, dp[index + 1] = max(dp[index + 1], dp[k + 1] + gain));
                    break;
                }
            }
        }
        return res;
    }

    vector<string> removeAnagrams(vector<string> &words) {
        int n = words.size();
        vector<string> res;
        res.push_back(words[0]);
        for (int i = 1; i < n; ++i) {
            string sortedWord = words[i];
            std::sort(sortedWord.begin(), sortedWord.end());
            string sortedPrevWord = words[i - 1];
            std::sort(sortedPrevWord.begin(), sortedPrevWord.end());
            if (sortedWord != sortedPrevWord) {
                res.push_back(words[i]);
            }
        }
        return res;
    }

    int magicalSum(int m, int k, vector<int> &nums) {
        constexpr int MOD = 1e9 + 7;
        int C[31][31] = {{0}};
        int n = nums.size();
        unordered_map<int, int> dp;
        auto modPow = [&](long long x, unsigned exp, int mod = MOD) -> long long {
            if (exp == 0) return 1;
            const int bM = 31 - countl_zero(exp);
            bitset<32> B(exp);
            long long y = x;
            for (int b = bM - 1; b >= 0; b--)
                y = (y * y % mod) * (B[b] ? x : 1) % mod;
            return y;
        };
        auto Pascal = [&]() {
            if (C[0][0] == 1) return;
            for (int i = 1; i <= 30; i++) {
                C[i][0] = C[i][i] = 1;
                for (int j = 1; j <= i / 2; j++) {
                    const int Cij = C[i - 1][j - 1] + C[i - 1][j];
                    C[i][j] = C[i][i - j] = Cij;
                }
            }
        };
        auto dfs = [&](auto &&dfs, int m, int k, int i, unsigned flag, vector<int> &nums) {
            const int bz = popcount(flag);
            if (m < 0 || k < 0 || m + bz < k)
                return 0;
            if (m == 0)
                return (k == bz) ? 1 : 0;
            if (i >= n) return 0;

            uint64_t key = (m << 5) | (k << 10) | (i << 16) | flag;
            auto it = dp.find(key);
            if (it != dp.end()) return it->second;

            long long ans = 0;
            for (int f = 0; f <= m; f++) {
                long long perm = C[m][f] * modPow(nums[i], f) % MOD;

                unsigned newFlag = flag + f;
                unsigned nextFlag = newFlag >> 1;
                bool bitContribution = newFlag & 1;

                ans = (ans + perm * dfs(dfs, m - f, k - bitContribution, i + 1, nextFlag, nums)) % MOD;
            }

            return dp[key] = ans;
        };
        Pascal();
        return dfs(dfs, m, k, 0, 0, nums);
    }

    bool hasIncreasingSubarrays(vector<int> &nums, int k) {
        int n = nums.size();
        if (k == 1) return true;
        int len = 1;
        int prev = 0;
        for (int i = 1; i < n; i++) {
            if (nums[i] > nums[i - 1]) len++;
            else {
                if (max(len / 2, min(len, prev)) >= k) return true;
                prev = len;
                len = 1;
            }
        }
        return max(len / 2, min(len, prev)) >= k;
    }

    int maxIncreasingSubarrays(vector<int> &nums) {
        int n = nums.size();
        int res = 1;
        int len = 1;
        int prev = 0;
        for (int i = 1; i < n; i++) {
            if (nums[i] > nums[i - 1]) len++;
            else {
                res = max(res, max(len / 2, min(len, prev)));
                prev = len;
                len = 1;
            }
        }
        return res = max(res, max(len / 2, min(len, prev)));
    }

    int findSmallestInteger(vector<int> &nums, int value) {
        vector<int> remains(value, 0);
        for (auto i: nums) {
            ++remains[i % value >= 0 ? i % value : i % value + value];
        }
        int res = std::numeric_limits<int>::max();
        for (int i = 0; i < value; ++i) {
            if (remains[i] > res) continue;
            res = min(remains[i] * value + i, res);
        }
        return res;
    }

    int maxPartitionsAfterOperations(const string &s, int k) {
        unordered_map<long long, int> cache;
        auto dp = [&](auto &&dp, long long index, long long currentSet, bool canChange) -> int {
            long long key = (index << 27) | (currentSet << 1) | canChange;
            if (cache.count(key)) {
                return cache[key];
            }
            if (index == s.size()) {
                return 0;
            }
            int characterIndex = s[index] - 'a';
            int currentSetUpdated = currentSet | (1 << characterIndex);
            int distinctCount = __builtin_popcount(currentSetUpdated);
            int res;
            if (distinctCount > k) {
                res = 1 + dp(dp, index + 1, 1 << characterIndex, canChange);
            } else {
                res = dp(dp, index + 1, currentSetUpdated, canChange);
            }
            if (canChange) {
                for (int newCharIndex = 0; newCharIndex < 26; ++newCharIndex) {
                    int newSet = currentSet | (1 << newCharIndex);
                    int newDistinctCount = __builtin_popcount(newSet);

                    if (newDistinctCount > k) {
                        res = max(res, 1 + dp(dp, index + 1, 1 << newCharIndex, false));
                    } else {
                        res = max(res, dp(dp, index + 1, newSet, false));
                    }
                }
            }
            return cache[key] = res;
        };
        return dp(dp, 0, 0, true) + 1;
    }

    int maxDistinctElements(vector<int> &nums, int k) {
        std::sort(nums.begin(), nums.end());
        int minNum = nums.front() - k;
        int res = 1;
        for (int i = 1; i < nums.size(); ++i) {
            if (minNum < nums[i] - k) {
                ++res;
                minNum = nums[i] - k;
            } else if (minNum < nums[i] + k) {
                ++minNum;
                ++res;
            }
        }
        return res;
    }

    string findLexSmallestString(const string &s, int a, int b) {
        int n = s.size();
        unordered_set<int> remainers;
        unordered_set<int> rotations;
        int tmp = b;
        while (!rotations.contains(tmp)) {
            rotations.insert(tmp);
            tmp = (tmp + b) % n;
        }
        tmp = a;
        while (!remainers.contains(tmp)) {
            remainers.insert(tmp);
            tmp = (tmp + a) % 10;
        }
        string res = s;
        for (int i: rotations) {
            string rotated = s.substr(i) + s.substr(0, i);
            for (int r: remainers) {
                string modified = rotated;
                for (int j = 1; j < n; j += 2) {
                    modified[j] = '0' + (modified[j] - '0' + r) % 10;
                }
                if (b % 2 == 0) {
                    res = min(res, modified);
                } else {
                    for (int r2: remainers) {
                        string modified2 = modified;
                        for (int j = 0; j < n; j += 2) {
                            modified2[j] = '0' + (modified2[j] - '0' + r2) % 10;
                        }
                        res = min(res, modified2);
                    }
                }
            }
        }
        return res;
    }

    int finalValueAfterOperations(vector<string> &operations) {
        int res = 0;
        for (const auto &op: operations) {
            if (op[1] == '+') {
                ++res;
            } else {
                --res;
            }
        }
        return res;
    }

    int maxFrequency(vector<int> &nums, int k, int numOperations) {
        if (nums.empty() || nums.size() == 1) return nums.size();
        vector<int> freq(100001, 0);
        for (auto i: nums) {
            ++freq[i];
        }
        if (numOperations == 0) {
            int res = 0;
            for (const auto &value: freq) {
                res = max(res, value);
            }
            return res;
        }

        std::sort(nums.begin(), nums.end());
        auto end = std::unique(nums.begin(), nums.end());
        nums.resize(std::distance(nums.begin(), end));
        int n = nums.size();
        int left = 0;
        int right = 0;
        int requiredOperations = freq[nums[0]];
        while (right + 1 < n && nums[right + 1] - nums[0] <= k) {
            right++;
            requiredOperations += freq[nums[right]];
        }
        int res = freq[nums[0]] + min(numOperations, requiredOperations - freq[nums[0]]);
        for (int pivot = nums.front() + 1; pivot <= nums.back(); ++pivot) {
            while (nums[left] < pivot && (pivot - nums[left]) > k) {
                requiredOperations -= freq[nums[left]];
                left++;
            }
            while (right + 1 < n && nums[right + 1] - pivot <= k) {
                requiredOperations += freq[nums[right + 1]];
                right++;
            }
            res = max(res, freq[pivot] + min(numOperations, requiredOperations - freq[pivot]));
        }
        return res;
    }

    bool hasSameDigits(string s) {
        int n = s.size();
        for (int i = 0; i < n - 2; ++i) {
            for (int j = 0; j < n - i - 1; ++j) {
                s[j] = (s[j] - '0' + s[j + 1] - '0') % 10 + '0';
            }
        }
        return s[0] == s[1];
    }

    int nextBeautifulNumber(int n) {
        vector<int> beautifuls = {1, 22, 122, 212, 221, 333, 1333, 3133, 3313, 3331, 4444, 14444, 22333, 23233, 23323, 23332, 32233, 32323, 32332, 33223, 33232, 33322, 41444, 44144, 44414, 44441, 55555, 122333, 123233, 123323, 123332, 132233, 132323, 132332, 133223, 133232, 133322, 155555, 212333, 213233, 213323, 213332, 221333, 223133, 223313, 223331, 224444, 231233, 231323, 231332, 232133, 232313, 232331, 233123, 233132, 233213, 233231, 233312, 233321, 242444, 244244, 244424, 244442, 312233, 312323, 312332, 313223, 313232, 313322, 321233, 321323, 321332, 322133, 322313, 322331, 323123, 323132, 323213, 323231, 323312, 323321, 331223, 331232, 331322, 332123, 332132, 332213, 332231, 332312, 332321, 333122, 333212, 333221, 422444, 424244, 424424, 424442, 442244, 442424, 442442, 444224, 444242, 444422, 515555, 551555, 555155, 555515, 555551, 666666, 1224444};
        return *std::upper_bound(beautifuls.begin(), beautifuls.end(), n);
    }

    int totalMoney(int n) {
        auto weekSum = [](int start) {
            return 7 * start + 21;
        };
        auto partWeekSum = [](int start, int days) {
            return days * start + (days * (days - 1)) / 2;
        };
        int fullWeeks = n / 7;
        int remainingDays = n % 7;
        int total = 0;
        total += (weekSum(1) + weekSum(fullWeeks)) * fullWeeks / 2 + partWeekSum(1 + fullWeeks, remainingDays);
        return total;
    }

    int numberOfBeams(vector<string> &bank) {
        int row = bank.size();
        int col = bank.front().size();
        int prevDevices = 0;
        int res = 0;
        for (int i = 0; i < row; ++i) {
            int currentDevices = std::count(bank[i].begin(), bank[i].end(), '1');
            if (currentDevices > 0) {
                res += prevDevices * currentDevices;
                prevDevices = currentDevices;
            }
        }
        return res;
    }

    int countValidSelections(vector<int> &nums) {
        int n = nums.size();
        int prefix = 0;
        int sum = std::accumulate(nums.begin(), nums.end(), 0);
        int res = 0;
        for (int i = 0; i < n; ++i) {
            prefix += nums[i];
            if (nums[i] == 0) {
                if (sum == prefix * 2) {
                    res += 2;
                }
                if (sum == prefix * 2 + 1 || sum == prefix * 2 - 1) {
                    res += 1;
                }
            }
        }
        return res;
    }

    int smallestNumber(int n) {
        for (int i = 0; i < 32; ++i) {
            if (((1 << i) - 1) >= n)
                return ((1 << i) - 1);
        }
        return ((1 << 10) - 1);
    }

    int minNumberOperations(vector<int> &target) {
        int n = target.size();
        int prev = 0;
        int res = 0;
        for (int i = 0; i < n; ++i) {
            if (target[i] > prev) {
                res += (target[i] - prev);
            }
            prev = target[i];
        }
        return res;
    }

    vector<int> getSneakyNumbers(vector<int> &nums) {
        vector<int> times(101, 0);
        vector<int> res;
        for (auto i: nums) {
            ++times[i];
            if (times[i] >= 2) {
                res.push_back(i);
            }
        }
        return res;
    }

    int minCost(string colors, vector<int> &neededTime) {
        int n = colors.size();
        int res = 0;
        int left = 0;
        while (left < n) {
            int right = left + 1;
            int maxTime = neededTime[left];
            int sumTime = neededTime[left];
            while (right < n && colors[right] == colors[left]) {
                sumTime += neededTime[right];
                maxTime = max(maxTime, neededTime[right]);
                ++right;
            }
            res += (sumTime - maxTime);
            left = right;
        }
        return res;
    }

    vector<int> findXSum(vector<int> &nums, int k, int x) {
        int n = nums.size();
        vector<int> res;
        res.reserve(n - k + 1);
        vector<int> numCount(51, 0);
        auto cmp = [&](auto &&a, auto &&b) {
            return numCount[a] > numCount[b] || (numCount[a] == numCount[b] && a > b);
        };
        priority_queue<int, vector<int>, decltype(cmp)> minHeap(cmp);
        auto calculateTopX = [&]() -> int {
            for (int i = 0; i <= 50; ++i) {
                minHeap.emplace(i);
                if (minHeap.size() > x) {
                    minHeap.pop();
                }
            }
            int ans = 0;
            while (!minHeap.empty()) {
                ans += minHeap.top() * numCount[minHeap.top()];
                minHeap.pop();
            }
            return ans;
        };
        for (int i = 0; i < k; ++i) {
            ++numCount[nums[i]];
        }
        res.push_back(calculateTopX());
        for (int i = 0; i + k < n; ++i) {
            numCount[nums[i]]--;
            numCount[nums[i + k]]++;
            res.push_back(calculateTopX());
        }
        return res;
    }

    vector<int> processQueries(int c, vector<vector<int>> &connections, vector<vector<int>> &queries) {

        vector<int> parent(c + 1, -1);
        vector<int> size(c + 1, 1);
        auto find = [&](auto &&find, int node) -> int {
            if (parent[node] == -1) {
                return node;
            }
            return parent[node] = find(find, parent[node]);
        };
        auto unionSet = [&](int u, int v) {
            int pu = find(find, u);
            int pv = find(find, v);
            if (pu != pv) {
                if (size[pu] < size[pv]) {
                    std::swap(pu, pv);
                }
                parent[pv] = pu;
                size[pu] += size[pv];
            }
        };
        for (const auto &conn: connections) {
            unionSet(conn[0], conn[1]);
        }
        unordered_map<int, set<int>> groups;
        for (int i = 1; i <= c; ++i) {
            groups[find(find, i)].insert(i);
        }
        size.clear();
        vector<int> res;
        res.reserve(queries.size());

        std::bitset<100001> isOnline;
        for (int i = 1; i <= c; ++i) {
            isOnline.set(i);
        }

        for (const auto &query: queries) {
            int type = query[0];
            int node = query[1];
            if (type == 1) {
                if (isOnline.test(node)) {
                    res.push_back(node);
                } else {
                    int pNode = find(find, node);
                    if (groups[pNode].empty()) {
                        res.push_back(-1);
                    } else {
                        res.push_back(*groups[pNode].begin());
                    }
                }
            } else {
                isOnline.reset(node);
                int pNode = find(find, node);
                groups[pNode].erase(node);
            }
        }
        return res;
    }

    long long maxPower(vector<int> &stations, int r, int k) {
        int n = stations.size();
        if (n == 0) return 0;
        int R = min(r, n - 1);
        long long mx = *max_element(stations.begin(), stations.end());
        long long right = mx * (2LL * R + 1) + k;
        long long left = 0, res = 0;
        long long prefix = 0;
        for (int j = 0; j < R; ++j) prefix += (long long) stations[j];
        auto can = [&](long long mid) -> bool {
            long long remain = (long long) k;
            long long slidingWindow = prefix;
            vector<long long> sanction(n, 0);
            for (int i = 0; i < n; ++i) {
                if (i - R - 1 >= 0) {
                    slidingWindow -= (long long) stations[i - R - 1];
                    slidingWindow -= sanction[i - R - 1];
                }
                if (i + R < n) {
                    slidingWindow += (long long) stations[i + R];
                    slidingWindow += sanction[i + R];
                }
                if (slidingWindow >= mid) continue;
                long long diff = mid - slidingWindow;
                if (remain < diff) return false;
                remain -= diff;
                int pos = min(n - 1, i + R);
                sanction[pos] += diff;
                slidingWindow += diff;
            }
            return true;
        };

        while (left <= right) {
            long long mid = (left + right) >> 1;
            if (can(mid)) {
                res = mid;
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return res;
    }

    int minimumOneBitOperations(unsigned int n) {
        if (n == 0) return 0;
        return ((1 << (32 - countl_zero(n))) - 1 - minimumOneBitOperations(n ^ bit_floor(n)));
    }

    int countOperations(int num1, int num2) {
        int res = 0;
        while (num1 != 0 && num2 != 0) {
            if (num1 >= num2) {
                num1 -= num2;
            } else {
                num2 -= num1;
            }
            ++res;
        }
        return res;
    }

    int minOperations(vector<int> &nums) {
        auto gcd = [](int a, int b) {
            while (b) {
                int temp = a % b;
                a = b;
                b = temp;
            }
            return a;
        };
        int n = nums.size();
        int overallGCD = nums[0];
        for (int i = 1; i < n; ++i) {
            overallGCD = gcd(overallGCD, nums[i]);
        }
        int minToOne = numeric_limits<int>::max();
        if (overallGCD > 1) return -1;
        for (int i = 0; i < n; ++i) {
            if (nums[i] == 1) {
                return n - std::count(nums.begin(), nums.end(), 1);
            }
            int localGCD = nums[i];
            if (minToOne <= 1) break;
            for (int j = i; j < n; ++j) {
                int prev = localGCD;
                localGCD = gcd(localGCD, nums[j]);
                if (localGCD == 1) {
                    minToOne = min(minToOne, j - i);
                    break;
                }
            }
        }
        return minToOne + n - 1 - std::count(nums.begin(), nums.end(), 1);
    }

    int maxOperations(const string &s) {
        int prev = -1;
        int prefix = 0;
        int res = 0;
        for (auto i: s) {
            if (i == '0' && prev != '0') {
                res += prefix;
            } else if (i == '1') {
                ++prefix;
            }
            prev = i;
        }
        return res;
    }

    vector<vector<int>> rangeAddQueries(int n, vector<vector<int>> &queries) {
        vector<vector<int>> matrix(n, vector<int>(n, 0));
        for (const auto &query: queries) {
            auto [row1, col1, row2, col2] = std::tuple<int, int, int, int>{query[0], query[1], query[2], query[3]};
            for (int i = row1; i <= row2; ++i) {
                matrix[i][col1] += 1;
                if (col2 + 1 < n) {
                    matrix[i][col2 + 1] -= 1;
                }
            }
        }
        vector<vector<int>> res(n, vector<int>(n, 0));
        for (int i = 0; i < n; ++i) {
            int prefix = 0;
            for (int j = 0; j < n; ++j) {
                prefix += matrix[i][j];
                res[i][j] = prefix;
            }
        }
        return res;
    }

    int numSub(const string &s) {
        int n = s.size();
        int firstOne = -1;
        const int MOD = 1e9 + 7;
        long long res = 0;
        for (int i = 0; i < n; ++i) {
            if (s[i] == '1') {
                if (firstOne == -1) {
                    firstOne = i;
                    res = (res + 1) % MOD;
                } else {
                    res = (res + (long long) (i - firstOne + 1)) % MOD;
                }
            } else {
                firstOne = -1;
            }
        }
        return res;
    }

    bool kLengthApart(vector<int> &nums, int k) {
        int distance = 0;
        for (int i = 0; i < nums.size(); ++i) {
            if (nums[i] == 1) {
                if (distance > 0) {
                    return false;
                }
                distance = k;
            } else {
                --distance;
            }
        }
        return true;
    }

    bool isOneBitCharacter(vector<int> &bits) {
        int n = bits.size();
        int i = 0;
        for (; i < n - 1; ++i) {
            if (bits[i] == 1) {
                ++i;
            }
        }
        return i != n;
    }

    int findFinalValue(vector<int> &nums, int original) {
        unordered_set<int> set(nums.begin(), nums.end());
        while (set.contains(original)) {
            original <<= 1;
        }
        return original;
    }

    int countPalindromicSubsequence(const string &s) {
        int n = s.size();
        vector<vector<int>> prefix(26, vector<int>(n + 1, 0));
        for (int i = 0; i < n; ++i) {
            for (int c = 0; c < 26; ++c) {
                prefix[c][i + 1] = prefix[c][i];
            }
            ++prefix[s[i] - 'a'][i + 1];
        }
        int res = 0;
        for (int c = 0; c < 26; ++c) {
            if (prefix[c].back() >= 2) {
                if (prefix[c].back() > 2)
                    ++res;
                auto firstPos = s.find_first_of('a' + c);
                auto lastPos = s.find_last_of('a' + c);
                for (int cc = 0; cc < 26; ++cc) {
                    if (cc == c) {
                        continue;
                    }
                    if (prefix[cc][lastPos + 1] - prefix[cc][firstPos] > 0) {
                        ++res;
                    }
                }
            } else if (prefix[c].back() < 2) {
                continue;
            }
        }
        return res;
    }

    int intersectionSizeTwo(vector<vector<int>> &intervals) {
        int n = intervals.size();
        std::sort(intervals.begin(), intervals.end(), [](const auto &a, const auto &b) {
            return a[1] != b[1] ? a[1] < b[1] : a[0] > b[0];
        });
        int res = 0;
        int first = -1;
        int second = -1;
        for (const auto &interval: intervals) {
            int start = interval[0];
            int end = interval[1];
            if (start <= first) {
                continue;
            }
            if (start > second) {
                res += 2;
                second = end;
                first = end - 1;
            } else {
                res += 1;
                first = second;
                second = end;
            }
        }
        return res;
    }

    int minimumOperations(vector<int> &nums) {
        return std::count_if(nums.begin(), nums.end(), [](int num) { return num % 3 != 0; });
    }

    int maxSumDivThree(vector<int> &nums) {
        int sum = 0;
        int min1 = INT_MAX, min2 = INT_MAX;
        int min11 = INT_MAX, min22 = INT_MAX;
        for (int x: nums) {
            sum += x;
            int r = x % 3;
            if (r == 1) {
                if (x < min1) {
                    min11 = min1;
                    min1 = x;
                } else if (x < min11)
                    min11 = x;
            } else if (r == 2) {
                if (x < min2) {
                    min22 = min2;
                    min2 = x;
                } else if (x < min22)
                    min22 = x;
            }
        }

        int rem = sum % 3;
        if (rem == 0) return sum;
        if (rem == 1) {
            int r1 = min1;
            int r2 = (min2 == INT_MAX || min22 == INT_MAX) ? INT_MAX : min2 + min22;
            int remove = min(r1, r2);
            return remove == INT_MAX ? 0 : sum - remove;
        } else {
            int r1 = min2;
            int r2 = (min1 == INT_MAX || min11 == INT_MAX) ? INT_MAX : min1 + min11;
            int remove = min(r1, r2);
            return remove == INT_MAX ? 0 : sum - remove;
        }
    }

    vector<bool> prefixesDivBy5(vector<int> &nums) {
        int n = nums.size();
        vector<bool> res(n, false);
        int prefix = 0;
        for (int i = 0; i < n; ++i) {
            prefix = ((prefix << 1) + nums[i]) % 5;
            res[i] = (prefix == 0);
        }
        return res;
    }

    int smallestRepunitDivByK(int k) {
        if (k == 1) return 1;
        if (k % 2 == 0) return -1;
        if (k % 5 == 0) return -1;
        int remain = 1 % k;
        int res = 1;
        std::bitset<100001> seen(0);
        while (remain != 0) {
            remain = (remain * 10 + 1) % k;
            ++res;
            if (seen.test(remain)) {
                return -1;
            } else {
                seen.set(remain);
            }
        }
        return res;
    }

    int numberOfPaths(vector<vector<int>> &grid, int k) {
        int n = grid.size();
        int m = grid[0].size();
        constexpr int MOD = 1e9 + 7;
        vector<vector<int>> dp(m, vector<int>(k, 0));
        vector<int> oldDp(k, 0);
        dp[0][grid[0][0] % k] = 1;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                if (i > 0) {
                    for (int rem = 0; rem < k; ++rem) {
                        oldDp[rem] = dp[j][rem];
                        dp[j][rem] = 0;
                    }
                    for (int rem = 0; rem < k; ++rem) {
                        int newRem = (rem + grid[i][j]) % k;
                        dp[j][newRem] = (oldDp[rem]) % MOD;
                    }
                }
                if (j > 0) {
                    for (int rem = 0; rem < k; ++rem) {
                        int newRem = (rem + grid[i][j]) % k;
                        dp[j][newRem] = (dp[j][newRem] + dp[j - 1][rem]) % MOD;
                    }
                }
            }
        }
        return dp[m - 1][0];
    }

    int maxKDivisibleComponents(int n, vector<vector<int>> &edges, vector<int> &values, int k) {
        if (n == 1) {
            return values[0] % k == 0 ? 1 : 0;
        }
        vector<vector<int>> graph(n);
        vector<int> indegree(n, 0);
        for (const auto &edge: edges) {
            graph[edge[0]].push_back(edge[1]);
            graph[edge[1]].push_back(edge[0]);
            ++indegree[edge[0]];
            ++indegree[edge[1]];
        }
        std::queue<int> q;
        for (int i = 0; i < n; ++i) {
            if (indegree[i] == 1) {
                q.push(i);
            }
        }
        int res = 0;
        vector<bool> visited(n, false);
        while (!q.empty()) {
            int node = q.front();
            q.pop();
            visited[node] = true;
            for (const auto &neighbor: graph[node]) {
                if (visited[neighbor]) continue;
                values[neighbor] += values[node];
                values[neighbor] %= k;
                --indegree[neighbor];
                if (indegree[neighbor] == 1) {
                    q.push(neighbor);
                }
            }
            if (values[node] % k == 0) {
                ++res;
            }
        }
        return res;
    }

    long long maxSubarraySum(vector<int> &nums, int k) {
        if (k == 1) {
            long long currentSum = 0;
            long long res = std::numeric_limits<long long>::min();
            for (int i = 0; i < nums.size(); ++i) {
                currentSum += nums[i];
                res = max(res, currentSum);
                if (currentSum < 0) {
                    currentSum = 0;
                }
            }
            return res;
        } else {
            vector<long long> prefixSums(k, 0);
            long long currentSum = 0;
            long long res = std::numeric_limits<long long>::min();
            for (int i = 0; i < k; ++i) {
                currentSum += nums[i];
                prefixSums[i] = currentSum;
            }
            res = max(res, currentSum);
            for (int i = k; i < nums.size(); ++i) {
                currentSum += nums[i];
                if ((i + 1) % k == 0) {
                    res = max(res, currentSum);
                }
                res = max(res, currentSum - prefixSums[i % k]);
                prefixSums[i % k] = min(prefixSums[i % k], currentSum);
            }
            return res;
        }
    }

    int minOperations(vector<int> &nums, int k) {
        return std::accumulate(nums.begin(), nums.end(), 0ll) % k;
    }

    int minSubarray(vector<int> &nums, int p) {
        int n = nums.size();
        int totalMod = 0;
        for (int i = 0; i < n; ++i) {
            totalMod = (totalMod + nums[i]) % p;
        }
        if (totalMod == 0) return 0;
        unordered_map<int, int> modIndex;
        int currentMod = 0;
        int res = n;
        for (int i = 0; i <= n; ++i) {
            currentMod = (i >= 1 ? (currentMod + nums[i - 1]) % p : 0);
            int needToFind = (currentMod - totalMod + p) % p;
            if (modIndex.find(needToFind) != modIndex.end()) {
                res = min(res, i - modIndex[needToFind]);
            }
            modIndex[currentMod] = i;
        }
        return res == n ? -1 : res;
    }

    long long maxRunTime(int n, vector<int> &batteries) {
        long long res = 0;
        if (batteries.size() < n) return 0;
        long long left = 1, right = std::accumulate(batteries.begin(), batteries.end(), 0ll) / n;
        while (left <= right) {
            long long mid = left + (right - left) / 2;
            long long total = 0;
            for (const auto &battery: batteries) {
                total += min((long long) battery, mid);
            }
            if (total >= mid * n) {
                res = mid;
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return res;
    }

    int countTrapezoids(vector<vector<int>> &points) {
        int n = points.size();
        constexpr int MOD = 1e9 + 7;
        long long res = 0;
        std::sort(points.begin(), points.end(), [](const auto &a, const auto &b) {
            return a[1] != b[1] ? a[1] < b[1] : a[0] < b[0];
        });
        int nowY = points[0][1];
        int count = 0;
        long long prefixSum = 0;
        for (int i = 1; i < n; ++i) {
            if (points[i][1] != nowY) {
                nowY = points[i][1];
                prefixSum += (long long) count * (count + 1) / 2 % MOD;
                count = 0;
            } else {
                ++count;
                res = (res + prefixSum * count % MOD) % MOD;
            }
        }
        return res;
    }

    int countCollisions(const string &directions) {
        int n = directions.size();
        int left = 0;
        int right = n - 1;
        while (left < n && directions[left] == 'L') {
            ++left;
        }
        while (right >= 0 && directions[right] == 'R') {
            --right;
        }
        int res = 0;
        for (int i = left; i <= right; ++i) {
            if (directions[i] != 'S') {
                ++res;
            }
        }
        return res;
    }

    int countPartitions(vector<int> &nums, int k) {
        const int MOD = 1e9 + 7;
        int n = nums.size();
        vector<int> dp(n + 1, 0);
        priority_queue<pair<int, int>> maxHeap;
        priority_queue<pair<int, int>, vector<pair<int, int>>, greater<>> minHeap;
        int start = 0;
        dp[0] = 1;
        int prefix = 0;
        for (int i = 0; i < n; ++i) {
            maxHeap.emplace(nums[i], i);
            minHeap.emplace(nums[i], i);
            if (maxHeap.top().first - minHeap.top().first > k) {
                while (maxHeap.top().first - minHeap.top().first > k) {
                    while (maxHeap.top().second < start) {
                        maxHeap.pop();
                    }
                    while (minHeap.top().second < start) {
                        minHeap.pop();
                    }
                    prefix = (prefix - dp[start] + MOD) % MOD;
                    ++start;
                    while (maxHeap.top().second < start) {
                        maxHeap.pop();
                    }
                    while (minHeap.top().second < start) {
                        minHeap.pop();
                    }
                }
            }
            prefix = (prefix + dp[i]) % MOD;
            dp[i + 1] = prefix;
        }
        return dp.back();
    }
};

int main() {
    [&out = std::cout]() -> void {
        out << "Hello World\n"
            << std::endl;
    }();
    auto isBeautiful = [](int num) {
        unordered_map<int, int> count;
        while (num) {
            count[num % 10]++;
            num /= 10;
        }
        for (const auto &[digit, freq]: count) {
            if (digit != freq) {
                return false;
            }
        }
        return true;
    };
    for (int i = 1; i <= 1224444; ++i) {
        if (isBeautiful(i)) {
            std::cout << i << ",";
        }
    }
    return 0;
}