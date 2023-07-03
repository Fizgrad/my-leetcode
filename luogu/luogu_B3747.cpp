//
// Created by david on 23-7-3.
//
#include <iostream>

using namespace std;

int main() {
    int n;
    cin >> n;
    cin.get();
    while (n--) {
        string x;
        int k;
        getline(cin, x);
        cin >> k;
        cin.get();
        if (x[0] == '0')
            printf("-1");
        else {
            for (int i = 1; i <= k; ++i) printf("9");
            for (int i = 1; i < x.length(); ++i) printf("0");
        }
        printf("\n");
    }
    return 0;
}