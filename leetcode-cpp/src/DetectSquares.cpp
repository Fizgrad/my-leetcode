//
// Created by David Chen on 4/2/23.
//
#include <algorithm>
#include <unordered_map>
#include <vector>

using namespace std;

class DetectSquares {
public:
    vector<unordered_map<int, int>> board;

    DetectSquares() : board(vector<unordered_map<int, int>>(1001)) {
    }

    void add(vector<int> point) {
        ++board[point[0]][point[1]];
    }

    int count(vector<int> point) {
        int res = 0;
        for (auto i = 0; i < 1001; ++i) {
            if (board[i][point[1]] > 0) {
                int len = abs(i - point[0]);
                if (len > 0) {
                    res += board[i][point[1] + len] * board[i][point[1]] *
                           board[point[0]][point[1] + len];
                    res += board[i][point[1] - len] * board[i][point[1]] *
                           board[point[0]][point[1] - len];
                }
            }
        }
        return res;
    }
};

int main() { return 0; }
/**
 * Your DetectSquares object will be instantiated and called as such:
 * DetectSquares* obj = new DetectSquares();
 * obj->add(point);
 * int param_2 = obj->count(point);
 */