#include <set>
#include <unordered_map>
#include <utility>

using namespace std;
class AllOne {
public:
    unordered_map<string, int> times;
    set<pair<int, string>> sorted;

    AllOne() {
    }

    void inc(const string &key) {
        int count = times[key];
        times[key]++;
        sorted.erase({count, key});
        sorted.insert({count + 1, key});
    }

    void dec(const string &key) {
        int count = times[key];
        times[key]--;
        sorted.erase({count, key});
        if (count > 1) { sorted.insert({count - 1, key}); }
    }

    string getMaxKey() {
        if (sorted.size())
            return sorted.rbegin()->second;
        else
            return "";
    }

    string getMinKey() {
        if (sorted.size())
            return sorted.begin()->second;
        else
            return "";
    }
};

/**
 * Your AllOne object will be instantiated and called as such:
 * AllOne* obj = new AllOne();
 * obj->inc(key);
 * obj->dec(key);
 * string param_3 = obj->getMaxKey();
 * string param_4 = obj->getMinKey();
 */

int main() {
    return 0;
}