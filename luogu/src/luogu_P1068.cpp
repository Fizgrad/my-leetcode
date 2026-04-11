//
// Created by david on 23-7-3.
//
#include <iostream>
#include <algorithm>

#define MAX_NUM 5000 + 5
using namespace std;

pair<int, int> people[MAX_NUM];

int main() {
    int n, m;
    cin >> n >> m;
    for (int i = 0; i < n; ++i) {
        cin >> people[i].second >> people[i].first;
    }
    sort(begin(people), end(people), [](pair<int, int> &a, pair<int, int> &b) {
        return a.first > b.first || (a.first == b.first && a.second < b.second);
    });
    int line = m + m / 2;
    int grade = people[line - 1].first;
    cout << grade << " ";
    int count = 0;
    for (int i = 0; i < n; ++i) {
        if (people[i].first >= grade)
            ++count;
    }
    cout << count << endl;
    for (int i = 0; i < count; ++i) {
        cout << people[i].second << " " << people[i].first << endl;
    }
    return 0;
}