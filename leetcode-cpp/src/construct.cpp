// Definition for a QuadTree node.
#include <iostream>
#include <vector>
using namespace  std;
class Node {
public:
    bool val;
    bool isLeaf;
    Node* topLeft;
    Node* topRight;
    Node* bottomLeft;
    Node* bottomRight;
    
    Node() {
        val = false;
        isLeaf = false;
        topLeft = NULL;
        topRight = NULL;
        bottomLeft = NULL;
        bottomRight = NULL;
    }
    
    Node(bool _val, bool _isLeaf) {
        val = _val;
        isLeaf = _isLeaf;
        topLeft = NULL;
        topRight = NULL;
        bottomLeft = NULL;
        bottomRight = NULL;
    }
    
    Node(bool _val, bool _isLeaf, Node* _topLeft, Node* _topRight, Node* _bottomLeft, Node* _bottomRight) {
        val = _val;
        isLeaf = _isLeaf;
        topLeft = _topLeft;
        topRight = _topRight;
        bottomLeft = _bottomLeft;
        bottomRight = _bottomRight;
    }
};


class Solution {
public:
    int all_equals(vector<vector<int>>& grid,int startx,int starty,int len){
        int temp = grid[startx][starty];
        for (int i = 0; i < len ;++i){
            for(int j = 0 ; j<len;++j){
                if(grid[startx+i][starty+j]==temp){
                    continue;
                }else {
                    return 0;
                }
            }
        }
        return temp == 1 ? 1:-1;
    }


    Node* dfs(vector<vector<int>>& grid,int start_x,int start_y,int len){
        if (len ==1){
            return new Node((grid[start_x][start_y]==1),true);
        }else {
            int value = all_equals(grid,start_x,start_y,len);
            if(value == 0){
                return new Node((grid[start_x][start_y]==1), false,
                                dfs(grid,start_x,start_y,len/2),
                                dfs(grid,start_x,start_y+len/2,len/2),
                                dfs(grid,start_x+len/2,start_y,len/2),
                                dfs(grid,start_x+len/2,start_y+len/2,len/2));
            }else {
                return new Node((value==1),true);
            }
        }
    }

    Node* construct(vector<vector<int>>& grid) {
        return dfs(grid,0,0,grid.size());

    }
};