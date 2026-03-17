
#include <vector>
using namespace std;

class Fancy {
public:
    static constexpr int mod = 1e9 + 7;
    long long add = 0, mult = 1;

    vector<long long> a;

    long long mod_add(long long a, long long b, long long m) {
        return ((a + b) % m + m) % m;
    }

    long long mod_mult(long long a, long long b, long long m) {
        return a * b % m;
    }

    long long binpow(long long a, long long b, long long m) {
        a %= m;
        long long res = 1;
        while (b) {
            if (b & 1) res = mod_mult(res, a, m);
            a = mod_mult(a, a, m);
            b >>= 1;
        }
        return res;
    }

    long long mod_inv(long long a, long long m) {
        return binpow(a, m - 2, m);
    }

    long long mod_div(long long a, long long b, long long m) {
        return a * mod_inv(b, m) % m;
    }

    Fancy() {
    }

    void append(int val) {
        a.push_back(mod_div(mod_add(val, -add, mod), mult, mod));
    }

    void addAll(int inc) {
        add = mod_add(add, inc, mod);
    }

    void multAll(int m) {
        add = mod_mult(add, m, mod);
        mult = mod_mult(mult, m, mod);
    }

    int getIndex(int idx) {
        return idx >= a.size() ? -1 : mod_add(mod_mult(a[idx], mult, mod), add, mod);
    }
};

int main() {
    return 0;
}