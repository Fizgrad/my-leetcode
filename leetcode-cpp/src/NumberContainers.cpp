#include <set>
#include <unordered_map>

class NumberContainers {
    std::unordered_map<int, std::set<int>> indices;
    std::unordered_map<int, int> maps;

public:
    NumberContainers() = default;

    void change(int index, int number) {
        if (maps.count(index)) {
            auto old = maps[index];
            indices[old].erase(index);
        }
        maps[index] = number;
        indices[number].emplace(index);
    }

    int find(int number) {
        if (indices[number].empty())
            return -1;
        else
            return *indices[number].begin();
    }
};

/**
     * Your NumberContainers object will be instantiated and called as such:
     * NumberContainers* obj = new NumberContainers();
     * obj->change(index,number);
     * int param_2 = obj->find(number);
     */

int main() {
    return 0;
}