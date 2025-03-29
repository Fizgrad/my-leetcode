#include <array>
#include <iostream>
#include <vector>
using namespace std;

constexpr int MAX_N = 1e5 + 5;
constexpr int MOD = 1e9 + 7;

auto is_prime = []() constexpr {
    std::array<bool, MAX_N> is_prime{};
    for (int i = 2; i < MAX_N; i++)
        is_prime[i] = true;

    for (int i = 2; i * i < MAX_N; i++) {
        if (is_prime[i]) {
            for (int j = i * i; j < MAX_N; j += i)
                is_prime[j] = false;
        }
    }
    return is_prime;
}();

auto fast_pow_mod = [](long long a, long long b, int MOD) {
    long long result = 1;
    a %= MOD;
    while (b > 0) {
        if (b % 2 == 1)
            result = (result * a) % MOD;
        a = (a * a) % MOD;
        b = b / 2;
    }
    return result;
};

auto primes = [](decltype(is_prime) &is_prime) {
    vector<int> primes;
    for (int i = 2; i < MAX_N; ++i) {
        if (is_prime[i]) {
            primes.emplace_back(i);
        }
    }
    return primes;
}(is_prime);

vector<int> primeScoresCache(MAX_N, -1);

auto calPrimeScore = [](int num, vector<int> &primeScoresCache) -> int {
    if (primeScoresCache[num] != -1) return primeScoresCache[num];
    int res = 0;
    int tmp = num;
    while (tmp > 1) {
        if (is_prime[tmp]) {
            ++res;
            return primeScoresCache[num] = res;
        }
        for (int i = 0; i < primes.size(); ++i) {
            if (tmp % primes[i] == 0) {
                ++res;
                while (tmp % primes[i] == 0) {
                    tmp /= primes[i];
                }
            }
            if (primes[i] > tmp)
                return primeScoresCache[num] = res;
        }
    }
    return primeScoresCache[num] = res;
};

class Solution {
public:
    int maximumScore(vector<int> &nums, long long k) {

        long long res = 1;

        auto getScore = [&](int i) {
            return calPrimeScore(nums[i], primeScoresCache);
        };

        priority_queue<pair<int, int>> pq;
        for (int i = 0; i < nums.size(); ++i) {
            pq.emplace(nums[i], i);
        }
        vector<int> front(nums.size(), -1);
        vector<int> back(nums.size(), -1);
        stack<int> monotonicStack;
        for (int i = 0; i < nums.size(); ++i) {
            while (!monotonicStack.empty() && getScore(monotonicStack.top()) < getScore(i)) {
                back[monotonicStack.top()] = i;
                monotonicStack.pop();
            }
            monotonicStack.emplace(i);
        }
        while (monotonicStack.size()) {
            monotonicStack.pop();
        }

        for (int i = nums.size() - 1; i >= 0; --i) {
            while (!monotonicStack.empty() && getScore(monotonicStack.top()) <= getScore(i)) {
                front[monotonicStack.top()] = i;
                monotonicStack.pop();
            }
            monotonicStack.emplace(i);
        }

        while (monotonicStack.size()) {
            monotonicStack.pop();
        }

        while (k > 0 && !pq.empty()) {
            auto [value, index] = pq.top();
            long long frontLen = index - front[index];
            long long backLen = back[index] - index;
            if (back[index] == -1) {
                backLen = nums.size() - index;
            }
            long long time = frontLen * backLen;
            res = (res * fast_pow_mod(value, min(time, k), MOD)) % MOD;
            k -= min(time, k);
            pq.pop();
        }

        return res;
    }
};

int main() {
    return 0;
}