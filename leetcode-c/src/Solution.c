#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

struct ListNode {
    int val;
    struct ListNode *next;
};

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

bool stoneGame(int *piles, int pilesSize) {
    int dp[505][505] = {0};
    for (int i = 0; i < pilesSize; i++) {
        dp[i][i] = piles[i];
    }
    for (int len = 2; len <= pilesSize; ++len) {
        for (int i = 0; i + len - 1 < pilesSize; ++i) {
            int j = i + len - 1;
            dp[i][j] = fmax(piles[i] - dp[i + 1][j], piles[i] - dp[i][j - 1]);
        }
    }
    return dp[0][pilesSize - 1] > 0;
}

char *stoneGameIII(int *stoneValue, int stoneValueSize) {
    int dp[50005] = {0};
    for (int i = stoneValueSize - 1; i >= 0; --i) {
        dp[i] = INT_MIN;
        int sum = 0;
        for (int k = 0; k < 3 && i + k < stoneValueSize; ++k) {
            sum += stoneValue[i + k];
            dp[i] = fmax(dp[i], sum - dp[i + k + 1]);
        }
    }
    if (dp[0] > 0) {
        return "Alice";
    } else if (dp[0] < 0) {
        return "Bob";
    } else {
        return "Tie";
    }
}

struct ListNode *reverseList(struct ListNode *dummy, struct ListNode *end) {
    struct ListNode *prev = end;// 关键修正：不是 dummy
    struct ListNode *cur = dummy->next;
    while (cur != end) {
        struct ListNode *next = cur->next;
        cur->next = prev;
        prev = cur;
        cur = next;
    }
    dummy->next = prev;// 让哨兵指向新头
    return prev;
}

struct ListNode *reverseKGroup(struct ListNode *head, int k) {
    if (k == 1) return head;
    if (!head) return head;
    struct ListNode *dummy = (struct ListNode *) malloc(sizeof(struct ListNode));
    dummy->next = head;
    struct ListNode *res = head;
    while (true) {
        struct ListNode *fast = dummy->next;
        struct ListNode *left = dummy->next;
        for (int i = 0; i < k; ++i) {
            if (fast == NULL) return res;
            fast = fast->next;
        }
        dummy = reverseList(dummy, fast);
        if (res == head) {
            res = dummy;
        }
        for (int i = 0; i < k - 1; ++i) {
            dummy = dummy->next;
        }
    }
}

int main() {
    int nums[] = {1, 5, 233, 7};
    int numsSize = sizeof(nums) / sizeof(nums[0]);
    predictTheWinner(nums, numsSize);
    return 0;
}