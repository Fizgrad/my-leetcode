#include <iostream>

using namespace std;

typedef long long ll;
const ll MOD = 998244353;
const ll PHI = MOD - 1; // 指数取模

struct Matrix {
    ll mat[2][2];
    Matrix() { mat[0][0] = mat[0][1] = mat[1][0] = mat[1][1] = 0; }
};

Matrix multiply(Matrix A, Matrix B) {
    Matrix C;
    for (int i = 0; i < 2; i++)
        for (int k = 0; k < 2; k++) // 优化循环顺序
            for (int j = 0; j < 2; j++)
                C.mat[i][j] = (C.mat[i][j] + A.mat[i][k] * B.mat[k][j]) % PHI;
    return C;
}

// 快速求斐波那契第 k 项 (F0=0, F1=1, F2=1, F3=2...)
ll get_fib(ll k) {
    if (k <= 0) return 0;
    if (k == 1) return 1;
    Matrix res, base;
    res.mat[0][0] = res.mat[1][1] = 1;
    base.mat[0][0] = 1; base.mat[0][1] = 1;
    base.mat[1][0] = 1; base.mat[1][1] = 0;
    
    ll p = k; 
    // 注意：这里矩阵幂次 A^p 作用于 [F1, F0] 会得到 [Fp+1, Fp]
    // 我们需要 Fk，所以算 A^(k-1) 后的 res.mat[0][0] 或是 A^k 后的 res.mat[0][1]
    p = k - 1;
    while (p > 0) {
        if (p & 1) res = multiply(res, base);
        base = multiply(base, base);
        p >>= 1;
    }
    return res.mat[0][0];
}

ll fast_pow(ll a, ll b) {
    ll res = 1;
    a %= MOD;
    while (b > 0) {
        if (b & 1) res = res * a % MOD;
        a = a * a % MOD;
        b >>= 1;
    }
    return res;
}

int main() {
    ll n;
    if (!(cin >> n)) return 0;

    if (n == 0) {
        cout << 1 << endl;
        return 0;
    }

    // 根据推导：
    // 底数 2 的总指数 Sa = F(n)
    // 底数 3 的总指数 Sb = F(n+1) - 1
    ll exp_a = get_fib(n) % PHI;
    ll exp_b = (get_fib(n + 1) - 1 + PHI) % PHI;

    ll term1 = fast_pow(2, exp_a);
    ll term2 = fast_pow(3, exp_b);

    cout << (term1 * term2 % MOD) << endl;

    return 0;
}