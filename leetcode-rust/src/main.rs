use std::collections::VecDeque;

// Definition for a binary tree node.
#[derive(Debug, PartialEq, Eq)]
pub struct TreeNode {
    pub val: i32,
    pub left: Option<Rc<RefCell<TreeNode>>>,
    pub right: Option<Rc<RefCell<TreeNode>>>,
}

impl TreeNode {
    #[inline]
    pub fn new(val: i32) -> Self {
        TreeNode {
            val,
            left: None,
            right: None,
        }
    }
}

use std::rc::Rc;
use std::cell::RefCell;

pub mod structs;

struct Solution {}

impl Solution {
    fn min_operations(nums: Vec<i32>, x: i32) -> i32 {
        let sum: i32 = nums.iter().sum();
        if sum < x {
            return -1;
        }
        let mut left = 0;
        let mut right = 0;
        let mut sum_left = 0;
        let mut sum_right = 0;
        let mut res = std::vec![];
        while sum_left < x {
            sum_left += nums[left];
            left += 1;
            if sum_left == x {
                res.push(left);
            }
        }
        while sum_right < x {
            sum_right += nums[nums.len() - (right + 1)];
            right += 1;
            while left > 0 && sum_right + sum_left > x {
                left -= 1;
                sum_left -= nums[left];
            }
            if sum_left + sum_right == x {
                res.push(left + right);
            }
        }
        return if res.is_empty() {
            -1
        } else {
            *res.iter().min().unwrap() as i32
        };
    }
    fn can_complete_circuit(gas: Vec<i32>, cost: Vec<i32>) -> i32 {
        let cost_all: i32 = cost.iter().sum();
        let gas_all: i32 = gas.iter().sum();
        if cost_all > gas_all {
            return -1;
        }
        let mut index = 0;
        let mut res = gas.len() - 1;
        let mut min = gas[0] - cost[0];
        let mut sum = 0;
        while index < gas.len() {
            sum += gas[index] - cost[index];

            index += 1;
            if sum < min {
                res = index;
                min = sum;
            }
        }
        return if res == gas.len() { 0 } else { res as i32 };
    }

    fn max_points(points: Vec<Vec<i32>>) -> i32 {
        fn gcd(first: i32, second: i32) -> i32 {
            let mut max = first;
            let mut min = second;
            if min == 0 || max == 0 {
                return 0;
            }

            if min > max {
                let val = max;
                max = min;
                min = val;
            }

            loop {
                let res = max % min;
                if res == 0 {
                    return min;
                }

                max = min;
                min = res;
            }
        }
        let n = points.len();
        if n <= 2 {
            return n as i32;
        }
        let mut ans = 0;
        for it1 in &points {
            let mut mp = std::collections::HashMap::new();
            mp.insert((0i32, i32::MAX), 0);
            let x1 = it1[0];
            let y1 = it1[1];
            for it2 in &points {
                if it2 == it1 {
                    continue;
                }
                let x2 = it2[0];
                let y2 = it2[1];
                let mut slope = (0i32, i32::MAX);
                if x2 - x1 == 0 {
                    // slope is infinity for vertical line
                } else {
                    let temp = gcd(y2 - y1, x2 - x1);
                    slope = if temp == 0 {
                        (y2 - y1, x2 - x1)
                    } else {
                        ((y2 - y1) / temp, (x2 - x1) / temp)
                    }
                }
                match mp.get(&slope) {
                    Some(num) => {
                        mp.insert(slope, num + 1);
                    }
                    None => {
                        mp.insert(slope, 1);
                    }
                }
                ans = i32::max(ans, mp.values().into_iter().max().unwrap().clone());
            }
        }
        return ans + 1; //including point i
    }

    fn product_except_self(nums: Vec<i32>) -> Vec<i32> {
        let num_zero = nums
            .iter()
            .filter(|x| !(x.is_negative() || x.is_positive()))
            .count();
        let len = nums.len();
        return if num_zero == 0 {
            let mut res = [0].repeat(len);
            let mut premuti = [nums[0]].repeat(len);
            let mut endmuti = [nums[len - 1]].repeat(len);
            let mut index = 1;
            while index < len {
                premuti[index] = nums[index] * premuti[index - 1];
                endmuti[len - 1 - index] = nums[len - index - 1] * endmuti[len - index];
                index += 1;
            }
            res[0] = endmuti[1];
            res[len - 1] = premuti[len - 2];
            index = 1;
            while index < len - 1 {
                res[index] = premuti[index - 1] * endmuti[index + 1];
                index += 1;
            }
            res
        } else if num_zero == 1 {
            let mut res = [0].repeat(len);
            let mut mult = 1;
            let mut index = 0;
            for (i, value) in nums.iter().enumerate() {
                if *value == 0 {
                    index = i;
                } else {
                    mult *= value;
                }
            }
            res[index] = mult;
            res
        } else {
            [0].repeat(len)
        };
    }

    fn min_moves2(nums: Vec<i32>) -> i32 {
        let mut nums = nums.clone();
        nums.sort();
        let mid: i32 = nums[nums.len() / 2];
        let mut res = 0;
        for i in nums {
            res += i32::abs(i - mid);
        }
        return res;
    }

    fn max_length_between_equal_characters(s: String) -> i32 {
        let mut res = 0;
        for (index1, value1) in s.char_indices() {
            for (index2, value2) in s.char_indices() {
                if index1 == index2 {
                    continue;
                } else {
                    if value1 == value2 {
                        res = i32::max(res, i32::abs((index1 - index2) as i32));
                    }
                }
            }
        }
        return res - 1;
    }

    fn reinitialize_permutation(n: i32) -> i32 {
        let mut nums = (0..n).collect::<Vec<i32>>();
        let mut tmp = nums.clone();
        let mut res = 0;
        let len = nums.len();
        loop {
            let a;
            let b;
            if res % 2 == 0 {
                a = &nums;
                b = &mut tmp;
            } else {
                a = &tmp;
                b = &mut nums;
            }
            for i in 0..len {
                b[i] = if i % 2 == 0 {
                    a[i / 2]
                } else {
                    a[len / 2 + (i - 1) / 2]
                }
            }
            res += 1;
            if b.iter().enumerate().all(|(a, b)| a as i32 == *b) {
                return res;
            }
        }
    }

    fn search_matrix(matrix: Vec<Vec<i32>>, target: i32) -> bool {
        let r = matrix.len();
        let c = matrix[0].len();
        let mut i: i64 = 0;
        let mut j: i64 = c as i64 - 1;
        while i > -1 && i < r as i64 && j > -1 && j < c as i64 {
            if target == matrix[i as usize][j as usize] {
                return true;
            }
            if target < matrix[i as usize][j as usize] {
                j -= 1;
                continue;
            }
            i += 1;
        }
        return false;
    }

    fn is_anagram(s: String, t: String) -> bool {
        let mut freqs = [0; 26];
        s.chars()
            .for_each(|c| freqs[c as usize - 'a' as usize] += 1);
        t.chars()
            .for_each(|c| freqs[c as usize - 'a' as usize] -= 1);
        freqs.iter().all(|&count| count == 0)
    }

    fn digit_count(num: String) -> bool {
        let mut hm = std::collections::HashMap::new();
        num.chars().into_iter().for_each(|v| {
            *hm.entry(v.to_digit(10).unwrap() as usize)
                .or_insert(0u32) += 1
        });
        println!("{:?}", hm);
        num.chars()
            .into_iter()
            .enumerate()
            .all(|(i, v)| *hm.entry(i).or_insert(0) == v.to_digit(10).unwrap())
    }

    fn min_time(n: i32, edges: Vec<Vec<i32>>, has_apple: Vec<bool>) -> i32 {
        use std::collections::HashMap;
        let g = edges
            .iter()
            .flat_map(|e| [[e[0], e[1]], [e[1], e[0]]])
            .fold(
                HashMap::<i32, Vec<i32>>::with_capacity(n as usize),
                |mut h, e| match h.get_mut(&e[0]) {
                    Some(list) => {
                        list.push(e[1]);
                        h
                    }
                    None => {
                        h.insert(e[0], vec![e[1]]);
                        h
                    }
                },
            );
        fn dfs(
            g: &HashMap<i32, Vec<i32>>,
            has_apple: &Vec<bool>,
            node: i32,
            parent: i32,
            depth: i32,
        ) -> i32 {
            match g
                .get(&node)
                .map(|nbrs| {
                    nbrs.iter()
                        .filter(|&&nb| nb != parent)
                        .map(|&child| dfs(g, has_apple, child, node, 1))
                        .sum()
                })
                .unwrap_or(0)
            {
                sum if sum > 0 || has_apple[node as usize] => 2 * depth + sum,
                _ => 0,
            }
        }
        return dfs(&g, &has_apple, 0, 0, 0);
    }

    fn zero_filled_subarray(nums: Vec<i32>) -> i64 {
        let mut num_zero = std::vec![];
        let mut temp = 0;
        nums.iter().for_each(|&v| {
            if v == 0 {
                temp += 1;
            } else {
                if temp != 0 {
                    num_zero.push(temp);
                    temp = 0;
                }
            }
        });
        return ((temp + 1) * temp) / 2 + num_zero.iter().map(|&x| (x + 1) * x / 2).sum::<i64>();
    }

    fn longest_path(parent: Vec<i32>, s: String) -> i32 {
        use std::collections::BinaryHeap;
        fn get_longest_path(
            graph: &Vec<Vec<usize>>,
            s: &[u8],
            root: usize,
            max_count: &mut i32,
        ) -> i32 {
            let root_char = s[root];
            let mut costs_heap = BinaryHeap::new();
            for &adj in graph[root].iter() {
                let sub_cost = get_longest_path(graph, s, adj, max_count);
                if root_char != s[adj] {
                    costs_heap.push(sub_cost);
                };
            }
            match costs_heap.pop() {
                Some(mut v0) => {
                    v0 += 1;
                    *max_count = if let Some(v1) = costs_heap.peek() {
                        (v0 + *v1).max(*max_count)
                    } else {
                        v0.max(*max_count)
                    };
                    v0
                }
                None => 1,
            }
        }
        let mut graph = vec![vec![]; parent.len()];
        let mut root = 0;
        for (child, &par) in parent.iter().enumerate() {
            if par == -1 {
                root = child;
                continue;
            }
            graph[par as usize].push(child);
        }
        let mut max_count = 1;
        get_longest_path(&graph, s.as_bytes(), root, &mut max_count);
        max_count
    }

    fn smallest_equivalent_string(s1: String, s2: String, base_str: String) -> String {
        fn find(tar: char, par: &Vec<char>) -> char {
            let mut res1 = tar;
            let mut res2 = par[res1 as usize - 'a' as usize];
            while res1 != res2 {
                res1 = res2;
                res2 = par[res1 as usize - 'a' as usize];
            }
            return res1;
        }

        fn union(a: char, b: char, par: &mut Vec<char>) {
            let par_a = find(a, par);
            let par_b = find(b, par);
            if par_a < par_b {
                par[par_b as usize - 'a' as usize] = par_a;
            } else if par_b < par_a {
                par[par_a as usize - 'a' as usize] = par_b;
            }
        }
        let mut par = ('a'..='z').collect::<Vec<char>>();
        let s2 = s2.chars().collect::<Vec<char>>();
        for (i, v) in s1.chars().into_iter().enumerate() {
            union(v, s2[i], &mut par);
        }
        let mut output = base_str.chars().collect::<Vec<char>>();
        for v in output.iter_mut() {
            *v = find(*v, &par);
        }
        return output.into_iter().collect::<String>();
    }
    fn count_different_subsequence_gc_ds(nums: Vec<i32>) -> i32 {
        fn gcd(first: i32, second: i32) -> i32 {
            if first == 0 {
                return second;
            }
            let mut max = first;
            let mut min = second;
            if min == 0 || max == 0 {
                return 0;
            }
            if min > max {
                let val = max;
                max = min;
                min = val;
            }
            loop {
                let res = max % min;
                if res == 0 {
                    return min;
                }
                max = min;
                min = res;
            }
        }
        const LIMIT: i32 = 200001;
        let mut ans = 0;
        let mut hash: Vec<bool> = std::vec![false; LIMIT as usize];
        nums.iter().for_each(|&num| {
            hash[num as usize] = true;
        });
        for i in 1..LIMIT {
            let mut g = 0;
            let mut j = i;
            while j < LIMIT {
                if hash[j as usize] {
                    g = gcd(g, j);
                }
                if g == i {
                    ans += 1;
                    break;
                }
                j += i;
            }
        }
        return ans;
    }

    fn word_pattern(pattern: String, s: String) -> bool {
        use std::collections::HashMap;
        let mut dict_pattern: HashMap<char, &str> = HashMap::new();
        let words = s.split(" ").collect::<Vec<&str>>();
        let mut idx = 0;
        if words.len() != pattern.len() {
            return false;
        }
        for i in pattern.chars() {
            if dict_pattern.contains_key(&i) {
                if dict_pattern.get(&i).unwrap().to_string() != words[idx].to_string() {
                    return false;
                }
            } else {
                if dict_pattern
                    .values()
                    .any(|x| x.to_string() == words[idx].to_string())
                {
                    return false;
                }
                dict_pattern.insert(i, words[idx]);
            }
            idx += 1;
        }
        true
    }
    fn number_of_good_paths(vals: Vec<i32>, edges: Vec<Vec<i32>>) -> i32 {
        fn find(y: &mut Vec<usize>, i: usize) -> usize {
            if i == y[i] {
                i
            } else {
                let mut yy = y.clone();
                let yi = find(&mut yy, y[i]);
                y.iter_mut().enumerate().for_each(|(i, v)| *v = yy[i]);
                y[i] = yi;
                yi
            }
        }
        let n = vals.len();
        let m = edges.len();
        let mut ans = 0;
        let mut x = std::vec![Vec::new(); n];
        let mut y = std::vec![0; n];
        for i in 0..n {
            y[i] = i;
            x[i] = std::vec![vals[i], 1];
        }
        let mut edges = edges.clone();
        edges.sort_by(|a, b| a.iter().max().cmp(&b.iter().max()));
        for i in 0..m {
            let a = find(&mut y, edges[i][0] as usize);
            let b = find(&mut y, edges[i][1] as usize);
            if x[a][0] != x[b][0] {
                if x[a][0] > x[b][0] {
                    y[b] = a;
                } else {
                    y[a] = b;
                }
            } else {
                y[a] = b;
                ans += x[a][1] * x[b][1];
                x[b][1] += x[a][1];
            }
        }
        return ans + n as i32;
    }
    // fn bisect(target: i32, array: &Vec<Vec<i32>>, index: usize) -> usize {
    //     let mut low = 0;
    //     let mut high = array.len();
    //     let mut mid = (low + high) / 2;
    //     while low <= high {
    //         match array[mid][index].cmp(&target) {
    //             std::cmp::Ordering::Equal => {
    //                 return mid;
    //             }
    //             std::cmp::Ordering::Greater => {
    //                 high = mid - 1;
    //                 mid = (low + high) / 2;
    //             }
    //             std::cmp::Ordering::Less => {
    //                 low = mid + 1;
    //                 mid = (low + high) / 2;
    //             }
    //         }
    //     }
    //     return mid;
    // }
    fn insert(intervals: Vec<Vec<i32>>, new_interval: Vec<i32>) -> Vec<Vec<i32>> {
        let mut ans: Vec<Vec<i32>> = std::vec![];
        let mut new_interval = new_interval.clone();
        for i in intervals {
            if i[1] < new_interval[0] {
                ans.push(i);
            } else if new_interval[1] < i[0] {
                ans.push(new_interval);
                new_interval = i;
            } else {
                new_interval[0] = i32::min(new_interval[0], i[0]);
                new_interval[1] = i32::max(new_interval[1], i[1]);
            }
        }
        ans.push(new_interval);
        return ans;
    }

    fn min_flips_mono_incr(s: String) -> i32 {
        s.chars()
            .fold((0, 0), |(ones, x), c| match c {
                '0' => (ones, ones.min(x + 1)),
                _ => (ones + 1, x),
            })
            .1
    }

    fn count_nice_pairs(nums: Vec<i32>) -> i32 {
        use std::collections::HashMap;
        let mut hm = HashMap::new();
        for mut i in nums {
            let mut i_copy = i.clone();
            let mut times = 0;
            while i_copy >= 1 {
                i_copy /= 10;
                times += 1;
            }
            i_copy = i.clone();
            for j in 1..=times {
                i -= (i_copy % 10) * 10_i32.pow(times - j);
                i_copy /= 10;
            }
            *hm.entry(i).or_insert(0i64) += 1;
        }
        return hm
            .values()
            .into_iter()
            .fold(0, |sum: i64, &x| (sum + ((x - 1) * x / 2)) % 1000000007) as i32;
    }
    fn max_subarray_sum_circular(nums: Vec<i32>) -> i32 {
        let all_sum = nums.iter().sum::<i32>();
        let mut max_min = (0, 0);
        let mut temp = (0, 0);
        for i in 0..nums.len() {
            temp = ((temp.0 + nums[i]).max(0), (temp.1 + nums[i]).min(0));
            max_min = (max_min.0.max(temp.0), max_min.1.min(temp.1));
        }
        if max_min.1 == 0 {
            return all_sum;
        }
        if max_min.0 == 0 {
            return *nums.iter().max().unwrap();
        }
        return max_min.0.max(all_sum - max_min.1);
    }

    fn subarrays_div_by_k(nums: Vec<i32>, k: i32) -> i32 {
        let mut remains_num = std::vec![0; k as usize];
        let mut sum = 0;
        for i in nums {
            sum += i;
            remains_num[(((sum % k) + k) % k) as usize] += 1;
        }
        remains_num[0] + remains_num.iter().fold(0, |a, &b| a + b * (b - 1) / 2)
    }
    fn finding_users_active_minutes(logs: Vec<Vec<i32>>, k: i32) -> Vec<i32> {
        use std::collections::HashMap;
        use std::collections::HashSet;
        let mut ans = std::vec![0; k as usize];
        let mut user_time = HashMap::new();
        for i in logs {
            user_time.entry(i[0]).or_insert(HashSet::new()).insert(i[1]);
        }
        for i in user_time.values() {
            ans[i.len() - 1] += 1;
        }
        ans
    }

    fn restore_ip_addresses(s: String) -> Vec<String> {
        let mut ans = std::vec![];
        if s.len() < 4 || s.len() > 12 {
            return ans;
        }
        fn dfs(depth: usize, s: &String, num_of_dot: i32, ans: &mut Vec<String>, path: Vec<u32>) {
            if num_of_dot == 3 {
                if s.len() - depth == 0 || s.len() - depth > 3 {
                    return;
                }
                if s.len() - depth != 1 && s.chars().nth(depth).unwrap() == '0' {
                    return;
                }
                let temp: u32 = s[depth..].parse().unwrap();
                if temp > 255 {
                    return;
                } else {
                    let mut new_string = String::new();
                    for i in path {
                        new_string += i.to_string().as_str();
                        new_string += '.'.to_string().as_str();
                    }
                    new_string += temp.to_string().as_str();
                    ans.push(new_string);
                }
            } else {
                for index in 0..=2 {
                    if depth < s.len() {
                        if index != 0 && s.chars().nth(depth).unwrap() == '0' {
                            continue;
                        }
                        let mut path = path.clone();
                        if depth + index < s.len() {
                            let temp_i = s[depth..=(depth + index)].parse::<u32>().unwrap();
                            if temp_i <= 255 {
                                path.push(temp_i);
                                dfs(depth + index + 1, s, num_of_dot + 1, ans, path);
                            }
                        }
                    }
                }
            }
        }
        dfs(0, &s, 0, &mut ans, std::vec![]);
        return ans;
    }

    fn partition(s: String) -> Vec<Vec<String>> {
        fn is_palindrome(s: &String) -> bool {
            let chars = s.chars().collect::<Vec<char>>();
            let n = chars.len();
            for i in 0..=(n / 2) {
                if chars[i] != chars[n - 1 - i] {
                    return false;
                }
            }
            return true;
        }
        let mut ans = std::vec![];
        fn partition_procedure(s: &String, ans: &mut Vec<Vec<String>>, dot: Vec<usize>) {
            let mut temp_index = 0;
            if dot.len() > 0 {
                temp_index = dot.last().unwrap() + 1;
                if *dot.last().unwrap() == s.len() - 1 {
                    let mut index = 0;
                    let mut temp_vec = std::vec![];
                    for i in dot.iter() {
                        temp_vec.push(s[index..=*i].to_string());
                        index = i + 1;
                    }
                    ans.push(temp_vec);
                }
            }
            for i in temp_index..=(s.len() - 1) {
                if is_palindrome(&s[temp_index..=i].to_string()) {
                    let mut temp = dot.clone();
                    temp.push(i);
                    partition_procedure(&s, ans, temp);
                }
            }
        }
        partition_procedure(&s, &mut ans, std::vec![]);
        return ans;
    }

    fn find_judge(n: i32, trust: Vec<Vec<i32>>) -> i32 {
        let mut hs = std::collections::HashSet::new();
        for i in trust.iter() {
            hs.insert(i[0]);
        }
        for i in 1..=n {
            if hs.contains(&i) {
                continue;
            }
            let mut hs_new = std::collections::HashSet::new();
            for j in trust.iter() {
                if j[1] == i {
                    hs_new.insert(j[0]);
                }
            }
            let mut flag = true;
            for k in 1..=n {
                if hs_new.contains(&k) || k == i {
                    continue;
                } else {
                    flag = false;
                }
            }
            if flag {
                return i;
            }
        }
        return -1;
    }
    fn snakes_and_ladders(board: Vec<Vec<i32>>) -> i32 {
        fn get_board(index: usize, board: &Vec<Vec<i32>>) -> i32 {
            let n = board.len();
            let row = (index + n - 1) / n;
            let column;
            if row % 2 == 0 {
                column = (n - index % n) % n;
            } else {
                column = (index + n - 1) % n;
            }
            return board[n - row][column];
        }
        let n = board.len();
        // let mut res = std::vec![i32::MAX;n*n+1];
        // res[1] = 0;
        // for i in 1..n * n + 1 {
        //     for k in 1..=6 {
        //         if i + k <= n * n {
        //             let next = get_board(i + k, &board);
        //             if next == -1 {
        //                 res[i + k] = res[i + k].min(res[i] + 1);
        //             } else {
        //                 res[next as usize] = res[next as usize].min(res[i] + 1);
        //             }
        //         } else {
        //             break;
        //         }
        //     }
        // }
        // return res[n * n];
        let mut moves = 0;
        let mut visited = std::vec![false; n * n + 1];
        let mut q = VecDeque::new();
        q.push_back(1);
        visited[1] = true;
        while !q.is_empty() {
            let size = q.len();
            for _ in 0..size {
                let curr_board_val = *q.front().unwrap();
                q.pop_front();
                if curr_board_val == n * n {
                    return moves;
                }
                for dice_val in 1..=6 {
                    if curr_board_val + dice_val > n * n {
                        break;
                    }
                    if visited[curr_board_val + dice_val] == false {
                        visited[curr_board_val + dice_val] = true;
                        let next = get_board(curr_board_val + dice_val, &board);
                        if next == -1 {
                            q.push_back(curr_board_val + dice_val);
                        } else {
                            q.push_back(next as usize);
                        }
                    }
                }
            }
            moves += 1;
        }
        return -1;
    }
    fn find_cheapest_price(n: i32, flights: Vec<Vec<i32>>, src: i32, dst: i32, k: i32) -> i32 {
        // I don't think dijstra algorithm would work here.
        // Take a look at this test case:
        // 5
        // [[0,1,5],[1,2,5],[0,3,2],[3,1,2],[1,4,1],[4,2,1]]
        // start: 0
        // goal: 2
        // Maximum stops: 2

        // The fastest way would be to go from 0-> 3-> 1-> 4-> 2. But this would have 3 stops, and 3 stops won't be allowed.
        // As a result we have to settle for 0->5->1->4->2. But if we're to use dijstra, this method will not be discovered because in dijstra only the fastest options are considered. But in this case, an actual dijstra doesn't consider it since the path is slower, so it would never find this solution, instead it'd be skipped over since it is less than the option that we already have.

        // As a result it is not really dijstra because due to it having to reinsert the same already explored nodes into the heap again, it'd run a lot slower.

        // Can you guys tell me if I'm right or wrong? Thanks.
        // Actually it is a slight variation in Dijkstra's. The distance that you calculate from the source has less priority than the number of stops and if you can build heap/PQ around num of stops, you can figure it out. (PS - as number of stops increase by +1 everytime, no need of PQ but just a normal Q suffices).
        use std::collections::HashMap;
        let mut hm = HashMap::new();
        for i in flights {
            hm.entry(i[0]).or_insert(std::vec![]).push((i[1], i[2]));
        }
        let mut cost = vec![i32::MAX; n as usize];
        cost[src as usize] = 0;
        for _ in 0..=k {
            let mut cost2 = cost.clone();
            for i in 0..n as usize {
                if cost[i] != i32::MAX {
                    let temp = hm.get(&(i as i32));
                    if temp.is_some() {
                        for &(j, price) in hm.get(&(i as i32)).unwrap() {
                            cost2[j as usize] = cost2[j as usize].min(cost[i] + price);
                        }
                    }
                }
            }
            cost = cost2;
        }
        match cost[dst as usize] {
            i32::MAX => -1,
            x => x,
        }
    }

    fn closest_meeting_node(edges: Vec<i32>, node1: i32, node2: i32) -> i32 {
        use std::collections::{HashSet, VecDeque};
        let len_vs: usize = edges.len();
        let mut queue1: VecDeque<i32> = VecDeque::with_capacity(len_vs);
        queue1.push_back(node1);
        let mut seen1: HashSet<i32> = HashSet::with_capacity(len_vs);
        seen1.insert(node1);
        let mut queue2: VecDeque<i32> = VecDeque::with_capacity(len_vs);
        queue2.push_back(node2);
        let mut seen2: HashSet<i32> = HashSet::with_capacity(len_vs);
        seen2.insert(node2);
        while !queue1.is_empty() || !queue2.is_empty() {
            let mut res: Vec<i32> = Vec::new();
            let len_q1: usize = queue1.len();
            for _ in 0..len_q1 {
                if let Some(cur) = queue1.pop_front() {
                    if seen2.contains(&cur) {
                        res.push(cur);
                    }
                    let nxt = edges[cur as usize];
                    if nxt != -1 && seen1.insert(nxt) {
                        queue1.push_back(nxt);
                    }
                }
            }
            let len_q2: usize = queue2.len();
            for _ in 0..len_q2 {
                if let Some(cur) = queue2.pop_front() {
                    if seen1.contains(&cur) {
                        res.push(cur);
                    }
                    let nxt = edges[cur as usize];
                    if nxt != -1 && seen2.insert(nxt) {
                        queue2.push_back(nxt);
                    }
                }
            }
            if !res.is_empty() {
                return res.into_iter().min().unwrap();
            }
        }
        -1
    }

    fn find_all_concatenated_words_in_a_dict(words: Vec<String>) -> Vec<String> {
        let mut res = std::vec![];
        let mut hs = std::collections::HashSet::new();
        words.iter().for_each(|i| {
            hs.insert(i);
        });
        for word in words.iter() {
            let n = word.len();
            let mut dp = std::vec![false; n + 1];
            dp[0] = true;
            for i in 0..n {
                if !dp[i] {
                    continue;
                }
                for j in i + 1..=n {
                    if j - i < n && hs.contains(&word[i..j].to_string()) {
                        dp[j] = true;
                    }
                }
                if dp[n] {
                    res.push(word.clone());
                    break;
                }
            }
        }
        return res;
    }
    fn merge(mut intervals: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
        intervals.sort_by_key(|k| k[0]);
        let mut ans: Vec<Vec<i32>> = std::vec![];
        let mut new_interval = intervals[0].clone();
        for i in intervals {
            if i[1] < new_interval[0] {
                ans.push(i);
            } else if new_interval[1] < i[0] {
                ans.push(new_interval);
                new_interval = i;
            } else {
                new_interval[0] = i32::min(new_interval[0], i[0]);
                new_interval[1] = i32::max(new_interval[1], i[1]);
            }
        }

        ans.push(new_interval);
        return ans;
    }

    fn ship_within_days(weights: Vec<i32>, days: i32) -> i32 {
        fn can_ship_within_days(weights: &Vec<i32>, capacity: i32, days: i32) -> bool {
            let mut day = 1;
            let mut weight = 0;
            for &i in weights {
                if weight + i > capacity {
                    weight = i;
                    day += 1;
                } else {
                    weight += i;
                }
                if day > days {
                    return false;
                }
            }
            return true;
        }
        let mut max_weight: i32 = weights.iter().sum();
        let mut min_weight = *weights.iter().max().unwrap();
        let mut mid = (max_weight + min_weight) / 2;

        while max_weight > min_weight {
            mid = (max_weight + min_weight) / 2;
            if can_ship_within_days(&weights, mid, days) {
                max_weight = mid;
            } else {
                min_weight = mid + 1;
            }
        }
        return max_weight;
    }
}

impl Solution {
    pub fn merge_similar_items(items1: Vec<Vec<i32>>, items2: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
        let mut hm = std::collections::BTreeMap::new();
        for i in items1 {
            hm.insert(i[0], i[1]);
        }
        for i in items2 {
            *hm.entry(i[0]).or_insert(0) += i[1];
        }
        let mut res = std::vec![];
        for i in hm {
            res.push(std::vec![i.0, i.1]);
        }

        return res;
    }
}

impl Solution {
    pub fn find_maximized_capital(
        mut k: i32,
        mut w: i32,
        profits: Vec<i32>,
        capital: Vec<i32>,
    ) -> i32 {
        let n = profits.len();
        let mut x: Vec<(i32, i32)> = (0..n).map(|x| (profits[x], capital[x])).collect();
        x.sort_unstable_by_key(|t| (*t).1);
        let mut ptr = 0;
        let mut bp = std::collections::BinaryHeap::new();
        while k > 0 {
            while ptr < n && x[ptr].1 <= w {
                bp.push(x[ptr].0);
                ptr += 1;
            }
            if let Some(tp) = bp.pop() {
                w += tp;
                k -= 1;
            } else {
                break;
            }
        }
        w
    }
}

impl Solution {
    pub fn circular_permutation(n: i32, start: i32) -> Vec<i32> {
        let mut res = std::vec![];
        for i in 0..(1 << n) {
            res.push(start ^ i ^ i >> 1);
        }
        return res;
    }
}

impl Solution {
    pub fn minimum_deviation(nums: Vec<i32>) -> i32 {
        let mut bts = std::collections::BTreeSet::new();
        fn to_smallest(x: i32) -> (i32, usize) {
            let mut times = 0;
            let mut i = x;
            if i % 2 != 0 {
                times = 1;
            }
            while i % 2 == 0 {
                times += 1;
                i = i / 2
            }
            (i, times)
        }
        for i in nums.iter() {
            bts.insert(to_smallest(*i));
        }
        let max_value = bts.iter().last().unwrap().0;
        let mut min_cp = *bts.iter().next().unwrap();
        let mut res = max_value - min_cp.0;
        while min_cp.0 != max_value {
            if min_cp.1 == 0 {
                res = res.min(bts.iter().last().unwrap().0 - bts.iter().next().unwrap().0);
                break;
            } else {
                bts.remove(&min_cp);
                bts.insert((min_cp.0 * 2, min_cp.1 - 1));
            }
            min_cp = *bts.iter().next().unwrap();
            res = res.min(bts.iter().last().unwrap().0 - min_cp.0);
        }
        res
    }
}

impl Solution {
    pub fn minimum_swap(s1: String, s2: String) -> i32 {
        let mut num_xy = 0;
        let mut num_yx = 0;
        let s2_chars: Vec<char> = s2.chars().collect();
        let s1_chars: Vec<char> = s1.chars().collect();
        for (i, &v) in s1_chars.iter().enumerate() {
            if v == *s2_chars.get(i).unwrap() {
                continue;
            } else {
                if v == 'x' {
                    num_xy += 1;
                } else {
                    num_yx += 1;
                }
            }
        }
        return if (num_xy + num_yx) % 2 == 1 {
            -1
        } else {
            num_xy / 2 + num_yx / 2 + (num_xy % 2) * 2
        };
    }
}

impl Solution {
    pub fn max_profit(mut prices: Vec<i32>) -> i32 {
        let mut res = 0;
        let mut temp = i32::MAX;
        for &v in prices.iter() {
            res = res.max(v - temp);
            temp = temp.min(v);
        }
        return res;
    }
}

impl Solution {
    pub fn min_distance(mut word1: String, mut word2: String) -> i32 {
        let l1 = word1.len();
        let l2 = word2.len();
        let mut dp = std::vec![std::vec![0; l2 + 1]; l1 + 1];
        let a: Vec<char> = word1.chars().collect();
        let b: Vec<char> = word2.chars().collect();
        // Base cases
        // Initializing First row
        for i in 0..=l2 {
            dp[0][i] = i;
        }
        // Initializing First col
        for i in 0..=l1 {
            dp[i][0] = i;
        }
        for i in 1..=l1 {
            for j in 1..=l2 {
                if a.get(i - 1).unwrap() == b.get(j - 1).unwrap() {
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    dp[i][j] = 1 + usize::min(
                        dp[i - 1][j - 1], // replace
                        usize::min(
                            dp[i - 1][j], // delete
                            dp[i][j - 1],
                        ), // insert
                    );
                }
            }
        }
        return dp[l1][l2] as i32;
    }
}

impl Solution {
    pub fn sort_array(nums: Vec<i32>) -> Vec<i32> {
        let mut heap = std::collections::BinaryHeap::new();
        for i in nums {
            heap.push(-i);
        }

        let mut res = std::vec![];
        while heap.peek().is_some() {
            res.push(heap.pop().unwrap());
        }
        return res;
    }
}

impl Solution {
    pub fn largest_local(grid: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
        let mut res = std::vec![std::vec![0; grid.len() - 2]; grid.len() - 2];
        let dx = [-1, -1, -1, 1, 1, 1, 0, 0, 0];
        let dy = [1, 0, -1, 1, 0, -1, 1, 0, -1];
        for i in 0..res.len() {
            for j in 0..res.len() {
                let mut max_num = 0;
                for k in 0..9usize {
                    let grid_i = grid.get((1 + i as i32 + dx[k]) as usize);
                    if grid_i.is_some() {
                        let grid_i_j = grid_i.unwrap().get((1 + j as i32 + dy[k]) as usize);
                        if grid_i_j.is_some() {
                            max_num = max_num.max(*grid_i_j.unwrap());
                        }
                    }
                }
                *res.get_mut(i)
                    .unwrap()
                    .get_mut(j)
                    .unwrap() = max_num;
            }
        }
        return res;
    }
}

impl Solution {
    pub fn compress(chars: &mut Vec<char>) -> i32 {
        let mut prev = *chars.get(0).unwrap();
        let mut modified_input = std::vec![];
        let mut num = 0;
        for &i in chars.iter() {
            if i == prev {
                num += 1;
            } else {
                if num == 1 {
                    modified_input.push(prev);
                } else {
                    modified_input.push(prev);
                    let num_string = num.to_string();
                    for j in num_string.chars() {
                        modified_input.push(j);
                    }
                }
                num = 1;
                prev = i;
            }
        }
        if num == 1 {
            modified_input.push(prev);
        } else {
            modified_input.push(prev);
            let num_string = num.to_string();
            for j in num_string.chars() {
                modified_input.push(j);
            }
        }
        *chars = modified_input;
        return chars.len() as i32;
    }

    pub fn deck_revealed_increasing(deck: Vec<i32>) -> Vec<i32> {
        let mut n = deck.len();
        let mut res = std::vec![0; n];
        let mut sorted = deck.clone();
        sorted.sort();
        let mut deque = std::collections::VecDeque::with_capacity(n);
        for i in 0..n {
            deque.push_back(i);
        }
        let mut order = std::vec![0usize; n];
        for i in 0..n {
            order[i] = *deque.front().unwrap();
            deque.pop_front();
            if !deque.is_empty() {
                deque.push_back(*deque.front().unwrap());
                deque.pop_front();
            }
        }
        for i in 0..n {
            res[order[i]] = sorted[i];
        }
        res
    }

    pub fn remove_kdigits(num: String, k: i32) -> String {
        let n = num.len();
        let mut res = String::new();
        if k >= n as i32 {
            return String::from("0");
        }
        let mut num_chars: Vec<char> = num.chars().collect();
        let mut kk = k;
        let mut stack = std::collections::VecDeque::new();
        for ch in num_chars {
            while !stack.is_empty() && *stack.back().unwrap() > ch && kk > 0 {
                kk -= 1;
                stack.pop_back();
            }
            stack.push_back(ch);
        }
        while kk > 0 {
            kk -= 1;
            stack.pop_back();
        }
        loop {
            if !stack.is_empty() && *stack.front().unwrap() == '0' {
                stack.pop_front();
            } else {
                break;
            }
        }
        for i in stack {
            res.push(i);
        }
        if res.is_empty() {
            res.push('0');
        }
        res
    }

    pub fn trap(height: Vec<i32>) -> i32 {
        let n = height.len();
        let mut left = std::vec![0; n];
        let mut right = std::vec![0; n];
        let mut res = 0;
        if n == 1 {
            return 0;
        }
        let mut temp = height[0];
        for i in 1..n {
            temp = i32::max(temp, height[i - 1]);
            left[i] = temp;
        }

        let mut temp = height[n - 1];
        for i in (0..n - 1).rev() {
            temp = i32::max(temp, height[i + 1]);
            right[i] = temp;
        }
        for i in 0..n {
            res += i32::max(0, i32::min(right[i], left[i]) - height[i]);
        }
        res
    }

    pub fn sum_of_left_leaves(root: Option<Rc<RefCell<TreeNode>>>) -> i32 {
        fn dfs(node: &Option<Rc<RefCell<TreeNode>>>, is_left: bool) -> i32 {
            if let Some(n) = node {
                let node = n.borrow();
                if is_left && node.left.is_none() && node.right.is_none() { node.val } else { dfs(&node.left, true) + dfs(&node.right, false) }
            } else { 0 }
        }
        dfs(&root, false)
    }

    pub fn add_one_row(root: Option<Rc<RefCell<TreeNode>>>, val: i32, depth: i32) -> Option<Rc<RefCell<TreeNode>>> {
        if depth == 1 {
            Some(Rc::new(RefCell::new(TreeNode {
                val: val,
                left: root,
                right: None,
            })))
        } else {
            fn dfs(node: &Option<Rc<RefCell<TreeNode>>>, deep: i32, depth: i32, val: i32) {
                if deep == depth - 1 {
                    if let Some(n) = node {
                        let mut node = n.borrow_mut();
                        node.left = Some(Rc::new(RefCell::new(TreeNode {
                            val: val,
                            left: node.left.take(),
                            right: None,
                        })));
                        node.right = Some(Rc::new(RefCell::new(TreeNode {
                            val: val,
                            left: None,
                            right: node.right.take(),
                        })));
                    }
                } else {
                    if let Some(n) = node {
                        let node = n.borrow();
                        if node.left.is_some() {
                            dfs(&node.left, deep + 1, depth, val);
                        }
                        if node.right.is_some() {
                            dfs(&node.right, deep + 1, depth, val);
                        }
                    }
                }
            }
            dfs(&root, 1, depth, val);
            root
        }
    }

    pub fn smallest_from_leaf(root: Option<Rc<RefCell<TreeNode>>>) -> String {
        fn dfs(root: &Option<Rc<RefCell<TreeNode>>>, mut path: String, ans: &mut String) {
            if let Some(node) = root {
                let node_ref = node.borrow();
                let ch = char::from_u32((node_ref.val as u32) + ('a' as u32)).unwrap();
                path.insert(0, ch);

                if node_ref.left.is_none() && node_ref.right.is_none() {
                    if ans.is_empty() || path < *ans {
                        *ans = path.clone();
                    }
                }
                dfs(&node_ref.left, path.clone(), ans);
                dfs(&node_ref.right, path.clone(), ans);
            }
        }
        let mut ans = String::new();
        dfs(&root, String::new(), &mut ans);
        ans
    }
}

fn main() {
    // println!("{:?}", (12,124) > (12,121));
    // println!("{:?}",(11,124) > (12,121));
    // println!("{:?}",(11,124) > (12,125));
    // println!("{:?}",(13,120) > (12,121));
    println!("{:?}", Solution::partition("aab".to_string()));
    println!(
        "{:?}",
        Solution::restore_ip_addresses("25525511135".to_string())
    );
    println!("{:?}", Solution::min_flips_mono_incr("010110".to_string()));
    println!(
        "{:?}",
        Solution::word_pattern("abbc".to_string(), "dog cat cat fish".to_string())
    );
    println!(
        "{:?}",
        Solution::count_different_subsequence_gc_ds(std::vec![5, 15, 40, 5, 6])
    );
    println!(
        "{:?}",
        Solution::smallest_equivalent_string(
            "leetcode".to_string(),
            "programs".to_string(),
            "sourcecode".to_string(),
        )
    );
    println!(
        "{:?}",
        Solution::smallest_equivalent_string(
            "hello".to_string(),
            "world".to_string(),
            "hold".to_string(),
        )
    );
    println!(
        "{:?}",
        Solution::is_anagram(
            std::string::String::from("rat"),
            std::string::String::from("car"),
        )
    );
    println!("{:?}", Solution::reinitialize_permutation(100));
    println!(
        "{:?}",
        Solution::min_operations(std::vec![1, 1, 4, 2, 3], 5)
    );
    println!(
        "{:?}",
        Solution::can_complete_circuit(std::vec![1, 2, 3, 4, 5], std::vec![3, 4, 5, 1, 2])
    );
    println!(
        "{:?}",
        Solution::can_complete_circuit(std::vec![1, 2], std::vec![2, 1])
    );
    println!(
        "{:?}",
        Solution::product_except_self(std::vec![-1, 1, 0, -3, 3])
    );
    println!(
        "{:?}",
        Solution::max_points(std::vec![
            std::vec![1, 1],
            std::vec![3, 2],
            std::vec![5, 3],
            std::vec![4, 1],
            std::vec![2, 3],
            std::vec![1, 4],
        ])
    );
    println!("{:?}", Solution::sort_array(std::vec![5, 1, 1, 2, 0, 0]));
}
