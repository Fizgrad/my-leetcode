//
// Created by David Chen on 5/29/23.
//


class ParkingSystem {
public:
    int spaces[3];

    ParkingSystem(int big, int medium, int small) {
        spaces[0] = big;
        spaces[1] = medium;
        spaces[2] = small;
    }

    bool addCar(int carType) {
        if (spaces[carType - 1]) {
            --spaces[carType - 1];
            return true;
        } else {
            return false;
        }
    }
};

int main() { return 0; }
/**
 * Your ParkingSystem object will be instantiated and called as such:
 * ParkingSystem* obj = new ParkingSystem(big, medium, small);
 * bool param_1 = obj->addCar(carType);
 */