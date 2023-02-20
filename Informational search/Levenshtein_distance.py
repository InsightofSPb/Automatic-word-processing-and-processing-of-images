def leven(s1, s2):
    n, m = len(s1), len(s2)

    cur_r = range(n + 1)
    for i in range(1, m + 1):
        pr_row, cur_r = cur_r, [i] + [0] * n
        for j in range(1, n + 1):
            add, delete, change = pr_row[j] + 1, cur_r[j - 1] + 1, pr_row[j - 1]
            if s1[j - 1] != s2[i - 1]:
                change +=1
            cur_r[j] = min(add,delete, change)
    return cur_r[n]


print(leven(input(), input()))
