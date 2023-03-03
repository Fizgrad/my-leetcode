class Solution:
    def isValidSudoku(self, board: list[list[str]]) -> bool:
        def isValid(digits: list[str]) -> bool:
            digits = digits.copy()
            while digits.count('.'):
                digits.remove('.')
            return len(digits) == len(set(digits))
        flag = True
        for i in range(9):
            if not isValid(board[i]):
                flag = False
                break
            if not isValid([board[j][i] for j in range(0, 9)]):
                flag = False
                break
        dx = [0, 0, 0, 1, 1, 1, 2, 2, 2]
        dy = [0, 1, 2, 0, 1, 2, 0, 1, 2]
        for i in range(0, 9, 3):
            for j in range(0, 9, 3):
                temp = []
                for k in range(9):
                    temp.append(board[i + dx[k]][j + dy[k]])
                if not isValid(temp):
                    flag = False
                    break
        return flag
