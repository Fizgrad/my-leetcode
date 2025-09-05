
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
};

int main() {
    [&out = std::cout]() -> void {
        out << "Hello World\n"
            << std::endl;
    }();
    return 0;
}