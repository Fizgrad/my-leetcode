//
// Created by david on 23-7-3.
//
#include <iostream>

#define MAX_NUM 1000000
using namespace std;

static int next[MAX_NUM + 10];

int main() {
    int n, h;
    cin >> n;
    for (int i = 1; i <= n; ++i) {
        cin >> ::next[i];
    }
    cin >> h;
    while (h != 0) {
        cout << h << " ";
        h = ::next[h];
    }
    return 0;
}
