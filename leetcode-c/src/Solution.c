#include <stdbool.h>
#include <stdio.h>
#include <math.h>

bool predictTheWinner(int *nums, int numsSize) {
    int dp[20][20] = {0};
    for (int i = 0; i < numsSize; i++) {
        dp[i][i] = nums[i];
    }
    for (int len = 2; len <= numsSize; len++) {
        for (int i = 0; i <= numsSize - len; i++) {
            int j = i + len - 1;
            dp[i][j] = fmax(nums[i] - dp[i + 1][j], nums[j] - dp[i][j - 1]);
        }
    }
    return dp[0][numsSize - 1] >= 0;
}

int main() {
    int nums[] = {1, 5, 233, 7};
    int numsSize = sizeof(nums) / sizeof(nums[0]);
    predictTheWinner(nums, numsSize);
    return 0;
}