#include <string>
#include <vector>
using namespace std;

class Robot {
public:
    int width;
    int height;
    vector<int> pos;
    vector<int> next;
    int direction = 0;
    constexpr static int dx[4] = {1, 0, -1, 0};
    constexpr static int dy[4] = {0, 1, 0, -1};
    constexpr static string direction_string[4] = {"East", "North", "West", "South"};

    Robot(int width, int height) : width(width),
                                   height(height), pos(2, 0) {
    }

    void step(int num) {
        if (num % (2 * width + 2 * height - 4) == 0) {
            if (pos[0] == 0 && pos[1] == 0) {
                direction = 3;
                return;
            }
            if (pos[0] == 0 && pos[1] == height - 1) {
                direction = 2;
                return;
            }
            if (pos[0] == width - 1 && pos[1] == height - 1) {
                direction = 1;
                return;
            }
            if (pos[0] == width - 1 && pos[1] == 0) {
                direction = 0;
                return;
            }
            return;
        }
        num = num % (2 * width + 2 * height - 4);
        next = pos;
        next[0] += dx[direction] * num;
        next[1] += dy[direction] * num;
        if (next[0] >= 0 && next[0] < width && next[1] >= 0 && next[1] < height) {
            pos = next;
            return;
        }
        if (direction == 0) {
            int walk = min(width - 1 - pos[0], num);
            if (walk < num) {
                direction = (direction + 1) % 4;
                pos = {pos[0] + walk, pos[1]};
                step(num - walk);
            } else {
                pos = {pos[0] + walk, pos[1]};
            }
        } else if (direction == 1) {
            int walk = min(height - 1 - pos[1], num);
            if (walk < num) {
                direction = (direction + 1) % 4;
                pos = {pos[0], pos[1] + walk};
                step(num - walk);
            } else {
                pos = {pos[0], pos[1] + walk};
            }
        } else if (direction == 2) {
            int walk = min(pos[0], num);
            if (walk < num) {
                direction = (direction + 1) % 4;
                pos = {pos[0] - walk, pos[1]};
                step(num - walk);
            } else {
                pos = {pos[0] - walk, pos[1]};
            }
        } else {
            int walk = min(pos[1], num);
            if (walk < num) {
                direction = (direction + 1) % 4;
                pos = {pos[0], pos[1] - walk};
                step(num - walk);
            } else {
                pos = {pos[0], pos[1] - walk};
            }
        }
    }

    vector<int> getPos() {
        return this->pos;
    }

    string getDir() {
        return direction_string[direction];
    }
};

/**
 * Your Robot object will be instantiated and called as such:
 * Robot* obj = new Robot(width, height);
 * obj->step(num);
 * vector<int> param_2 = obj->getPos();
 * string param_3 = obj->getDir();
 */


int main() {
    return 0;
}