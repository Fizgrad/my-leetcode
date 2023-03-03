import java.util.*;

public class Solution {
     class TreeNode {
          int val;
          TreeNode left;
          TreeNode right;
          TreeNode() {}
          TreeNode(int val) { this.val = val; }
          TreeNode(int val, TreeNode left, TreeNode right) {
              this.val = val;
              this.left = left;
              this.right = right;
          }
    }
    public int bestTeamScore(int[] scores, int[] ages) {
        int N = ages.length;
        int[][] ageScorePair = new int[N][2];

        for (int i = 0; i < N; i++) {
            ageScorePair[i][0] = scores[i];
            ageScorePair[i][1] = ages[i];
        }

        // Sort in ascending order of score and then by age.
        Arrays.sort(ageScorePair, (a, b) -> a[0] == b[0] ? a[1] - b[1] : a[0] - b[0]);

        int highestAge = 0;
        for (int i : ages) {
            highestAge = Math.max(highestAge, i);
        }
        int[] BIT = new int[highestAge + 1];

        int answer = Integer.MIN_VALUE;
        for (int[] ageScore : ageScorePair) {
            // Maximum score up to this age might not have all the players of this age.
            int currentBest = ageScore[0] + queryBIT(BIT, ageScore[1]);
            // Update the tree with the current age and its best score.
            updateBIT(BIT, ageScore[1], currentBest);

            answer = Math.max(answer, currentBest);
        }

        return answer;
    }

    // Query tree to get the maximum score up to this age.
    private int queryBIT(int[] BIT, int age) {
        int maxScore = Integer.MIN_VALUE;
        for (int i = age; i > 0; i -= i & (-i)) {
            maxScore = Math.max(maxScore, BIT[i]);
        }
        return maxScore;
    }

    // Update the maximum score for all the nodes with an age greater than the given age.
    private void updateBIT(int[] BIT, int age, int currentBest) {
        for (int i = age; i < BIT.length; i += i & (-i)) {
            BIT[i] = Math.max(BIT[i], currentBest);
        }
    }


    public String gcdOfStrings(String str1, String str2) {
        if (str2.length() > str1.length())
            return gcdOfStrings(str2, str1);
        if (str2.equals(str1))
            return str1;
        if (str1.startsWith(str2))
            return gcdOfStrings(str1.substring(str2.length()), str2);
        return "";
    }

    public boolean compareString(HashMap<Character, Integer> map, String a, String b) {
        for (int j = 0; j < Integer.min(a.length(), b.length()); ++j) {
            int result = map.get(a.charAt(j)) - map.get(b.charAt(j));
            if (result > 0) {
                return false;
            } else if (result < 0) {
                return true;
            }
        }
        return true;
    }

    public boolean isAlienSorted(String[] words, String order) {
        HashMap<Character, Integer> map = new HashMap<>();
        for (int i = 0; i < order.length(); ++i) {
            map.put(order.charAt(i), i);
        }
        for (int i = 1; i < words.length; ++i) {
            if (!compareString(map, words[i - 1], words[i])) {
                return false;
            }
        }
        return true;
    }

    public String convert(String s, int numRows) {
        if (numRows == 1) {
            return s;
        }
        StringBuilder[] strings = new StringBuilder[numRows];
        for (int i = 0; i < s.length(); ++i) {
            int index = (i % (2 * numRows - 2));
            if (index >= numRows) {
                index = 2 * numRows - index - 2;
            }
            if (strings[index] == null) {
                strings[index] = new StringBuilder();
            }
            strings[index].append(s.charAt(i));
        }
        for (int j = numRows - 2; j >= 0; j--) {
            if (strings[j + 1] != null) {
                strings[j].append(strings[j + 1]);
            }
        }
        return strings[0].toString();
    }

    public boolean checkInclusion(String s1, String s2) {
        if (s1.length() > s2.length()) {
            return false;
        }
        int[] numOfChar = new int[26];
        for (int i = 0; i < s1.length(); ++i) {
            ++numOfChar[s1.charAt(i) - 'a'];
        }
        int i = 0;
        int j = i + s1.length();
        for (int k = 0; k < j; ++k) {
            --numOfChar[s2.charAt(k) - 'a'];
        }
        while (j < s2.length()) {
            if (Arrays.stream(numOfChar).allMatch((x) -> x == 0)) {
                return true;
            } else {
                ++numOfChar[s2.charAt(i) - 'a'];
                ++i;
                --numOfChar[s2.charAt(j) - 'a'];
                ++j;
            }
        }
        return Arrays.stream(numOfChar).allMatch((x) -> x == 0);
    }

    public ArrayList<Integer> findAnagrams(String s, String p) {
        ArrayList<Integer> ans = new ArrayList<>();
        if (p.length() > s.length()) {
            return ans;
        }
        int[] numOfChar = new int[26];
        for (int i = 0; i < p.length(); ++i) {
            ++numOfChar[p.charAt(i) - 'a'];
        }
        int i = 0;
        int j = i + p.length();
        for (int k = 0; k < j; ++k) {
            --numOfChar[s.charAt(k) - 'a'];
        }
        while (j < s.length()) {
            if (Arrays.stream(numOfChar).allMatch((x) -> x == 0)) {
                ans.add(i);
            } else {
                ++numOfChar[s.charAt(i) - 'a'];
                ++i;
                --numOfChar[s.charAt(j) - 'a'];
                ++j;
            }
        }
        if (Arrays.stream(numOfChar).allMatch((x) -> x == 0)) {
            ans.add(i);
        }
        return ans;
    }


    public int trap(int[] height) {
        int temp = 0;
        int[] resHeight = new int[height.length];
        for (int i = 0; i < height.length; ++i) {
            temp = Integer.max(temp, height[i]);
            resHeight[i] = temp;
        }
        temp = 0;
        for (int i = 0; i < height.length; ++i) {
            temp = Integer.max(temp, height[height.length - 1 - i]);
            resHeight[height.length - 1 - i] = Integer.min(temp, resHeight[height.length - 1 - i]);
        }
        return Arrays.stream(resHeight).sum() - Arrays.stream(height).sum();
    }

    public int totalFruit(int[] fruits) {
        if (fruits.length <= 2) {
            return fruits.length;
        }
        int res = 0;
        int[] type = {-1, -1};
        int[] count = new int[2];
        for (int fruit : fruits) {
            if (type[0] == -1) {
                type[0] = fruit;
                count[0] = 1;
            } else if (fruit == type[0]) {
                ++count[0];
            } else if (type[1] == -1) {
                type[1] = type[0];
                count[1] = count[0];
                type[0] = fruit;
                count[0] = 1;
            } else if (fruit == type[1]) {
                type[1] = type[0];
                type[0] = fruit;
                count[1] = count[0] + count[1];
                count[0] = 1;
            } else {
                count[1] = count[0];
                type[1] = type[0];
                type[0] = fruit;
                count[0] = 1;
            }
            System.out.print(type[0]);
            System.out.println(type[1]);
            System.out.print(count[0]);
            System.out.println(count[1]);

            res = Integer.max(res, count[0] + count[1]);
        }
        return res;

    }

    public int jump(int[] nums) {
        int end = 0, farthest = 0, count = 0;
        for (int i = 0; i < nums.length - 1; i++) {
            farthest = Integer.max(farthest, nums[i] + i);
            if (end == i) {
                end = farthest;
                count++;
            }
        }
        return count;
    }

    public long distinctNames(String[] ideas) {
        HashSet<String>[] hashSets = (HashSet<String>[]) new HashSet[26];
        for (int i = 0; i < hashSets.length; ++i) {
            hashSets[i] = new HashSet<>();
        }

        for (String i : ideas) {
            hashSets[i.charAt(0) - 'a'].add(i.substring(1));
        }
        long res = 0;
        for (int i = 0; i < 26; ++i) {
            for (int j = i + 1; j < 26; ++j) {
                int sizeOfi = hashSets[i].size();
                hashSets[i].addAll(hashSets[j]);
                res += (long) (sizeOfi - hashSets[i].size()) * (hashSets[j].size() - hashSets[i].size());
            }
        }
        return 2 * res;
    }

    public int maxDistance(int[][] grid) {
        class Coordinate {
            final int x;
            final int y;

            Coordinate(int x, int y) {
                this.x = x;
                this.y = y;
            }
        }
        int n = grid.length;
        boolean flagWater = false;
        Queue<Coordinate> queue = new LinkedList<>();
        boolean[][] visited = new boolean[n][n];
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (grid[i][j] == 1) {
                    queue.add(new Coordinate(i, j));
                    visited[i][j] = true;
                } else if (grid[i][j] == 0) {
                    flagWater = true;
                }
            }
        }
        if (queue.isEmpty() || !flagWater) {
            return -1;
        }
        int res = 0;
        int[] dx = {0, 0, 1, -1};
        int[] dy = {-1, 1, 0, 0};
        while (!queue.isEmpty()) {
            Queue<Coordinate> temp = new LinkedList<>();
            while (!queue.isEmpty()) {
                Coordinate top = queue.peek();
                queue.remove();
                for (int k = 0; k < 4; ++k) {
                    int xx = top.x + dx[k];
                    int yy = top.y + dy[k];
                    if (xx >= 0 && xx < n && yy >= 0 && yy < n) {
                        if (!visited[xx][yy]) {
                            temp.add(new Coordinate(xx, yy));
                            visited[xx][yy] = true;
                        }
                    }
                }
            }
            queue.addAll(temp);
            ++res;
        }
        return res;
    }

    public int[] shortestAlternatingPaths(int n, int[][] redEdges, int[][] blueEdges) {
        Set<Integer>[][] graph = new HashSet[2][n];
        for (int i = 0; i < n; i++) {
            graph[0][i] = new HashSet<>();
            graph[1][i] = new HashSet<>();
        }
        for (int[] red : redEdges) {
            graph[0][red[0]].add(red[1]);
        }
        for (int[] blue : blueEdges) {
            graph[1][blue[0]].add(blue[1]);
        }
        int[][] res = new int[2][n];
        for (int i = 1; i < n; i++) {
            res[0][i] = 2 * n;
            res[1][i] = 2 * n;
        }
        Queue<int[]> q = new LinkedList<>();
        q.offer(new int[]{0, 0});
        q.offer(new int[]{0, 1});
        while (!q.isEmpty()) {
            int[] current = q.poll();
            int vert = current[0];
            int color = current[1];
            for (int next : graph[1 - color][vert]) {
                if (res[1 - color][next] == 2 * n) {
                    res[1 - color][next] = 1 + res[color][vert];
                    q.offer(new int[]{next, 1 - color});
                }
            }
        }
        int[] ans = new int[n];
        for (int i = 0; i < n; i++) {
            int t = Math.min(res[0][i], res[1][i]);
            ans[i] = (t == 2 * n) ? -1 : t;
        }
        return ans;
    }

    int seats;
    long res = 0;

    public int minimumFuelCostDFS(int node, int prev, Vector<Vector<Integer>> nexts) {
        Vector<Integer> temp = nexts.get(node);
        int people = 1;
        for (Integer i : temp) {
            if (i == prev) continue;
            people += minimumFuelCostDFS(i, node, nexts);
        }
        if (node != 0) {
            res += (people + seats - 1) / seats;
        }
        return people;
    }

    public long minimumFuelCost(int[][] roads, int seats) {
        this.seats = seats;
        int n = roads.length + 1;
        Vector<Vector<Integer>> graph = new Vector<>(n);
        for (int i = 0; i < n; ++i) {
            graph.add(new Vector<>());
        }
        for (int[] i : roads) {
            graph.get(i[0]).add(i[1]);
            graph.get(i[1]).add(i[0]);
        }
        minimumFuelCostDFS(0, 0, graph);
        return res;
    }

    public int countOdds(int low, int high) {
        int n = high - low + 1;
        return (n % 2 == 0) ? (n / 2) : (n / 2 + (low % 2));
    }

    //1234
    public int balancedString(String s) {
        char[] arr = s.toCharArray();
        int[] freq = new int[128];
        for (char c : arr) freq[c]++;
        int l = 0, n = s.length(), k = n / 4, res = n;
        for (int r = 0; r < n; r++) {
            --freq[arr[r]];
            while (l < n && freq['W'] <= k && freq['E'] <= k && freq['R'] <= k && freq['Q'] <= k) {
                res = Math.min(res, r - l + 1);
                ++freq[arr[l++]];
            }
        }
        return res;
    }
    public List<Integer> addToArrayForm(int[] num, int k) {
        LinkedList<Integer> res = new LinkedList<>();
        int len = num.length;
        while(len > 0 || k != 0){
            if(len > 0){
                k += num[--len];
            }
            res.addFirst(k % 10);
            k /= 10;
        }
        return res;
    }
    Integer prev = null;
    int resMinDiffInBST = Integer.MAX_VALUE;
    public int minDiffInBST(TreeNode root) {
        if (root.left != null)
            minDiffInBST(root.left);

        if (prev != null)
            resMinDiffInBST = Math.min(resMinDiffInBST, root.val - prev);

        prev = root.val;

        if (root.right != null) minDiffInBST(root.right);
        return resMinDiffInBST;
    }
    public int largest1BorderedSquare(int[][] grid) {
        int n = grid.length;
        int m = grid[0].length;
        int[][] hor = new int[n][m];
        int[][] ver = new int[n][m];
        for(int i = 0; i< n;++i){
            for(int j = 0; j< m;++j){
                if(grid[i][j]==1) {
                    hor[i][j] = (j == 0) ? 1 : hor[i][j - 1] + 1;
                    ver[i][j] = (i == 0) ? 1 : ver[i - 1][j] + 1;
                }
            }
        }
        int res = 0;
        for(int i = n-1; i>= 0;--i){
            for(int j = m-1; j>= 0;--j){
                int min = Integer.min(hor[i][j],ver[i][j]);
                int length = min;
                while(length > res){
                    if(hor[i-length+1][j] >= length && ver[i][j-length+1]>=length) {
                        res = Integer.max(res, length);
                    }
                    length--;
                }
            }
        }
        return res*res;
    }
//    public boolean isSame(TreeNode a, TreeNode b){
//        return a.val == b.val && (a.left==null? b.left==null:b.left!=null&&isSame(a.left,b.left)) && (a.right==null? b.right==null:b.right!=null && isSame(a.right,b.right));
//    }
//
//    public boolean traversal(List<TreeNode> res, TreeNode root,TreeNode pattern){
//        if(root != null){
//            if (isSame(root,pattern)) {
//                res.add(root);
//                return true;
//            }else {
//                boolean result = false;
//                result = traversal(res,root.right,pattern) || result;
//                result = traversal(res,root.left,pattern) || result;
//                return  result;
//            }
//        }
//        return false;
//    }
    public List<TreeNode> findDuplicateSubtrees(TreeNode root) {
        List<TreeNode> res=new LinkedList<>();
        HashMap<String,Integer> hm=new HashMap<>();
        helper(res,root,hm);
        return res;
    }
    public StringBuilder helper(List<TreeNode> res,TreeNode root,HashMap<String,Integer> hm){
        if(root==null)
            return new StringBuilder();
        StringBuilder left=helper(res,root.left,hm);
        StringBuilder right=helper(res,root.right,hm);
        StringBuilder curr_root= new StringBuilder(Integer.toString(root.val));
        StringBuilder sb=curr_root.append('.');
        sb.append(left);
        sb.append('.');
        sb.append(right);
        String string_formed  = sb.toString();
        if(hm.getOrDefault(string_formed,0)==1){
            res.add(root);
        }
        hm.put(string_formed,hm.getOrDefault(string_formed,0)+1);
        return sb;
    }

    public static void main(String[] args) {
        String[] ideas = {"coffee", "donuts", "time", "toffee"};
        int [] nums = {1,2,0,0};
        Solution s = new Solution();
        System.out.println(s.distinctNames(ideas));
        System.out.println(s.addToArrayForm(nums,34));

    }


}