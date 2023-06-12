//
// Created by David Chen on 6/12/23.
//

#include <iostream>
#include <vector>

using namespace std;

class SnapshotArray {
public:
    vector<vector<pair<int, int>>> snapshot;
    int snap_id = 0;

    SnapshotArray(int length) : snapshot(length, vector<pair<int, int>>(1, {0, 0})) {

    }

    void set(int index, int val) {
        if (snapshot[index].back().first != snap_id) {
            snapshot[index].emplace_back(snap_id, val);
        } else {
            snapshot[index].back().second = val;
        }

    }

    int snap() {
        return snap_id++;
    }

    int get(int index, int snap_id) {
        auto &snaps = snapshot[index];
        int left = 0;
        int right = snaps.size() - 1;

        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (snaps[mid].first <= snap_id) {
                if (mid == snaps.size() - 1 || snaps[mid + 1].first > snap_id) {
                    return snaps[mid].second;
                } else {
                    left = mid + 1;
                }
            } else {
                right = mid - 1;
            }
        }

        return 0; // 如果找不到合适的快照，则返回默认值（此处为0）
    }

};

/**
 * Your SnapshotArray object will be instantiated and called as such:
 * SnapshotArray* obj = new SnapshotArray(length);
 * obj->set(index,val);
 * int param_2 = obj->snap();
 * int param_3 = obj->get(index,snap_id);
 */