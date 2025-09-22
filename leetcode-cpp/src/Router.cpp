#include <array>
#include <queue>
#include <unordered_map>
#include <vector>
using namespace std;

class Router {
    int size_;
    unordered_map<long long, array<int, 3>> packets;
    unordered_map<int, vector<int>> cnt;
    queue<long long> q;

    static long long encode(int s, int d, int t) {
        return (((long long) s << 40) | ((long long) d << 20) | (long long) t);
    }

public:
    Router(int memoryLimit) {
        size_ = memoryLimit;
        packets.reserve(max(1, memoryLimit * 2));
    }
    bool addPacket(int source, int destination, int timestamp) {
        long long key = encode(source, destination, timestamp);
        if (packets.count(key))
            return false;
        if ((int) packets.size() >= size_) {
            forwardPacket();
        }
        packets[key] = {source, destination, timestamp};
        q.push(key);

        auto &v = cnt[destination];
        auto it = lower_bound(v.begin(), v.end(), timestamp);
        v.insert(it, timestamp);
        return true;
    }
    vector<int> forwardPacket() {
        if (packets.empty())
            return {};
        long long key = q.front();
        q.pop();
        auto it = packets.find(key);
        if (it == packets.end())
            return {};
        auto packet = it->second;
        packets.erase(it);
        int d = packet[1];
        int ts = packet[2];
        auto &v = cnt[d];
        auto itv = lower_bound(v.begin(), v.end(), ts);
        if (itv != v.end() && *itv == ts)
            v.erase(itv);
        return {packet[0], packet[1], packet[2]};
    }
    int getCount(int destination, int startTime, int endTime) {
        auto it = cnt.find(destination);
        if (it == cnt.end())
            return 0;
        auto &v = it->second;
        auto L = lower_bound(v.begin(), v.end(), startTime);
        auto R = upper_bound(v.begin(), v.end(), endTime);
        return (int) (R - L);
    }
};

/**
 * Your Router object will be instantiated and called as such:
 * Router* obj = new Router(memoryLimit);
 * bool param_1 = obj->addPacket(source,destination,timestamp);
 * vector<int> param_2 = obj->forwardPacket();
 * int param_3 = obj->getCount(destination,startTime,endTime);
 */

int main() {
    return 0;
}