"""
Sudoku Solver - CSP Sprint Challenge
Difficulty: Easy
Interview companies: Google, Amazon, Microsoft

Your task: Complete the solve_sudoku() and is_valid() functions
"""

def print_board(board):
    """Prints the Sudoku board in a nice format"""
    for i in range(9):
        if i % 3 == 0 and i != 0:
            print("- - - - - - - - - - - -")
        for j in range(9):
            if j % 3 == 0 and j != 0:
                print(" | ", end="")
            if j == 8:
                print(board[i][j])
            else:
                print(str(board[i][j]) + " ", end="")

def find_empty_cell(board):
    """
    Finds the next empty cell (represented by 0)
    Returns: (row, col) tuple or None if no empty cells
    """
    for i in range(9):
        for j in range(9):
            if board[i][j] == 0:
                return (i, j)
    return None

def is_valid(board, row, col, num):
    """
    TODO: Implement this function!
    
    Check if placing 'num' at board[row][col] is valid
    
    A placement is valid if:
    1. 'num' is not already in the same row
    2. 'num' is not already in the same column
    3. 'num' is not already in the same 3x3 box
    
    Args:
        board: 9x9 2D list representing the Sudoku board
        row: row index (0-8)
        col: column index (0-8)
        num: number to place (1-9)
    
    Returns:
        True if valid placement, False otherwise
    
    Hints:
    - Check row: loop through board[row][:]
    - Check column: loop through board[:][col]
    - Check 3x3 box: find which box this cell is in
      Box row start = (row // 3) * 3
      Box col start = (col // 3) * 3
    """
    
    # TODO: Check if num is in the same row
    
    
    # TODO: Check if num is in the same column
    
    
    # TODO: Check if num is in the same 3x3 box
    
    
    return True  # Replace this with your logic

def solve_sudoku(board):
    """
    TODO: Implement this function!
    
    Solves the Sudoku puzzle using backtracking
    
    Algorithm:
    1. Find an empty cell (use find_empty_cell helper)
    2. If no empty cells, puzzle is solved! Return True
    3. For each number 1-9:
       a. Check if placing this number is valid (use is_valid helper)
       b. If valid, place the number
       c. Recursively try to solve the rest (call solve_sudoku again)
       d. If recursive call succeeds, return True
       e. If recursive call fails, UNDO the placement (backtrack!)
    4. If all numbers 1-9 fail, return False
    
    Args:
        board: 9x9 2D list (modified in place)
    
    Returns:
        True if solved, False if no solution exists
    """
    
    # TODO: Find empty cell
    empty = find_empty_cell(board)
    
    # TODO: If no empty cells, we're done!
    
    
    # TODO: Try numbers 1-9
    
    
    # TODO: If we tried all numbers and none worked, return False
    
    
    pass  # Remove this when you implement the function

# Test cases
if __name__ == "__main__":
    # Easy puzzle
    board1 = [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]
    ]
    
    print("Original Sudoku puzzle:")
    print_board(board1)
    print("\nSolving...")
    
    if solve_sudoku(board1):
        print("\nSolved!")
        print_board(board1)
    else:
        print("\nNo solution exists")
    
    # Medium puzzle
    board2 = [
        [0, 0, 0, 6, 0, 0, 4, 0, 0],
        [7, 0, 0, 0, 0, 3, 6, 0, 0],
        [0, 0, 0, 0, 9, 1, 0, 8, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 5, 0, 1, 8, 0, 0, 0, 3],
        [0, 0, 0, 3, 0, 6, 0, 4, 5],
        [0, 4, 0, 2, 0, 0, 0, 6, 0],
        [9, 0, 3, 0, 0, 0, 0, 0, 0],
        [0, 2, 0, 0, 0, 0, 1, 0, 0]
    ]
    
    print("\n" + "="*40)
    print("\nMedium puzzle:")
    print_board(board2)
    print("\nSolving...")
    
    if solve_sudoku(board2):
        print("\nSolved!")
        print_board(board2)
    else:
        print("\nNo solution exists")

"""
Expected output for board1:
5 3 4 | 6 7 8 | 9 1 2
6 7 2 | 1 9 5 | 3 4 8
1 9 8 | 3 4 2 | 5 6 7
- - - - - - - - - - - -
8 5 9 | 7 6 1 | 4 2 3
4 2 6 | 8 5 3 | 7 9 1
7 1 3 | 9 2 4 | 8 5 6
- - - - - - - - - - - -
9 6 1 | 5 3 7 | 2 8 4
2 8 7 | 4 1 9 | 6 3 5
3 4 5 | 2 8 6 | 1 7 9
"""
