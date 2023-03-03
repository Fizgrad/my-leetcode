#include <vector>
#include <iostream>
#include <string>

using namespace std;
class Solution {
public:
    const int dx[4] = {0,0,1,-1};
    const int dy[4] = {1,-1,0,0};
    bool next_procedure(vector<vector<char>>& board,string& word,int index,int x,int y,vector<vector<bool>>& checks) {
        ++index;
        if(index == word.size())
            return true;
        for(int i = 0;i<4;++i){
            if(x +dx[i]>=0 && x +dx[i]<board.size() && y+dy[i] >=0&&y+dy[i]<board.front().size()){
                if(!checks[x+dx[i]][y+dy[i]]&&board[x+dx[i]][y+dy[i]]==word[index]){
                    checks[x+dx[i]][y+dy[i]] = true;
                    if( next_procedure(board,word,index,x+dx[i],y+dy[i],checks)){
                        return true;
                    }
                    checks[x+dx[i]][y+dy[i]] = false;
                }
            }
        }
        return false;
    }

    bool exist(vector<vector<char>>& board, string word) {
        vector<bool> temp(10, false);
        vector<vector<bool>> checks(10, temp);
        for (int i = 0; i < board.size(); ++i) {
            for (int j = 0; j < board.front().size(); ++j) {
                if (board[i][j] == word.front()) {
                    checks[i][j] = true;
                    if (next_procedure(board, word, 0, i, j, checks)) {
                        return true;
                    }
                    checks[i][j] = false;
                }
            }
        }
        return false;
    }
};

int main(){

}