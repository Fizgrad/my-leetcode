#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


/*
 * 返回状态。
 */
typedef enum {
    KMP_SUCCESS = 0,       /* 找到匹配 */
    KMP_NOT_FOUND,         /* 未找到 */
    KMP_INVALID_ARGUMENT,  /* 参数不合法 */
    KMP_ALLOCATION_FAILED, /* 内存申请失败 */
    KMP_SIZE_OVERFLOW      /* 内存大小计算溢出 */
} KmpStatus;

/*
 * 在 haystack[0, haystack_len) 中查找 needle[0, needle_len)。
 *
 * 参数约束：
 * 1. result 不能为 NULL。
 * 2. haystack_len > 0 时，haystack 必须指向至少 haystack_len 字节的可读内存。
 * 3. needle_len > 0 时，needle 必须指向至少 needle_len 字节的可读内存。
 *
 * 成功找到时：
 *     返回 KMP_SUCCESS；
 *     *result 指向第一次匹配的位置。
 *
 * 未找到时：
 *     返回 KMP_NOT_FOUND；
 *     *result 为 NULL。
 *
 * 空模式串：
 *     认为在 haystack 开头匹配成功。
 *
 * 该接口显式传入长度，因此也可以搜索包含 '\0' 的二进制数据。
 */
KmpStatus kmp_find(
        const char *haystack,
        size_t haystack_len,
        const char *needle,
        size_t needle_len,
        const char **result) {
    size_t *lps = NULL;
    size_t i;
    size_t j;

    if (result == NULL) {
        return KMP_INVALID_ARGUMENT;
    }

    /* 先初始化输出，避免失败时留下未初始化值。 */
    *result = NULL;

    if ((haystack == NULL && haystack_len != 0) ||
        (needle == NULL && needle_len != 0)) {
        return KMP_INVALID_ARGUMENT;
    }

    /*
     * 空模式串按照 strstr 的语义，在主串开头匹配成功。
     * haystack_len == 0 时，haystack 允许为 NULL。
     */
    if (needle_len == 0) {
        *result = haystack;
        return KMP_SUCCESS;
    }

    if (haystack_len < needle_len) {
        return KMP_NOT_FOUND;
    }

    /*
     * 防止 needle_len * sizeof(*lps) 发生 size_t 溢出。
     */
    if (needle_len > SIZE_MAX / sizeof(*lps)) {
        return KMP_SIZE_OVERFLOW;
    }

    lps = malloc(needle_len * sizeof(*lps));
    if (lps == NULL) {
        return KMP_ALLOCATION_FAILED;
    }

    /*
     * 构造 LPS 数组。
     *
     * lps[i] 表示：
     * needle[0..i] 的最长“真前缀且同时为后缀”的长度。
     *
     * 例如模式串 "abab"：
     *
     * 下标： 0 1 2 3
     * 字符： a b a b
     * lps：  0 0 1 2
     */
    lps[0] = 0;

    i = 1;
    j = 0;

    while (i < needle_len) {
        if (needle[i] == needle[j]) {
            ++j;
            lps[i] = j;
            ++i;
        } else if (j != 0) {
            /*
             * 不移动 i，继续尝试更短的相等前后缀。
             */
            j = lps[j - 1];
        } else {
            lps[i] = 0;
            ++i;
        }
    }

    /*
     * 执行 KMP 匹配。
     *
     * i：主串当前位置
     * j：模式串当前位置
     */
    i = 0;
    j = 0;

    while (i < haystack_len) {
        if (haystack[i] == needle[j]) {
            ++i;
            ++j;

            if (j == needle_len) {
                *result = haystack + (i - j);
                free(lps);
                return KMP_SUCCESS;
            }
        } else if (j != 0) {
            /*
             * 主串位置 i 不回退，只调整模式串位置。
             */
            j = lps[j - 1];
        } else {
            ++i;
        }
    }

    free(lps);
    return KMP_NOT_FOUND;
}

KmpStatus kmp_strstr(
        const char *haystack,
        const char *needle,
        const char **result) {
    if (result == NULL) {
        return KMP_INVALID_ARGUMENT;
    }

    *result = NULL;

    if (haystack == NULL || needle == NULL) {
        return KMP_INVALID_ARGUMENT;
    }

    return kmp_find(
            haystack,
            strlen(haystack),
            needle,
            strlen(needle),
            result);
}


int main(void) {
    const char *haystack = "abcxabcdabxabcdabcdabcy";
    const char *needle = "abcdabcy";
    const char *match = NULL;

    KmpStatus status = kmp_strstr(haystack, needle, &match);

    switch (status) {
        case KMP_SUCCESS:
            printf("找到：%s\n", match);
            printf("起始下标：%td\n", match - haystack);
            break;

        case KMP_NOT_FOUND:
            printf("未找到\n");
            break;

        case KMP_INVALID_ARGUMENT:
            fprintf(stderr, "参数不合法\n");
            break;

        case KMP_ALLOCATION_FAILED:
            fprintf(stderr, "内存申请失败\n");
            break;

        case KMP_SIZE_OVERFLOW:
            fprintf(stderr, "模式串长度过大\n");
            break;
    }

    return status == KMP_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}