def analyzeboard(board):
    cb=[[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]]

    for i in range(0,8):
        if(board[cb[i][0]] != 0 and
           board[cb[i][0]] == board[cb[i][1]] and
           board[cb[i][0]] == board[cb[i][2]]):
            return board[cb[i][2]]
    return 0

def minimax(board,player):
    x=analyzeboard(board)
    if(x!=0):
        return (x*player)
    pos=-1
    value=-2
    for i in range(0,9):
        if(board[i]==0):
            print_board(board)
            print("player X" if player==-1 else "Player O")
            board[i]=player
            score=-minimax(board,(player*-1))
            print(f"pos = {i}, score = {score}")
            if(score>value):
                value=score
                pos=i
            board[i]=0

    if(pos==-1):
        return 0
    return value

def CompTurn(board):
    pos=-1
    value=-2
    for i in range(0,9):
        if(board[i]==0):
            print_board(board)
            print("player O")
            board[i]=1
            score=-minimax(board, -1)
            board[i]=0
            if(score>value):
                value=score
                pos=i
 
    return pos + 1 # change to 1 indexing 

def print_board(vector):
    """
    Convert a vector of 9 elements to a tic-tac-toe board display.
    
    Args:
        vector (list): List of 9 elements where -1 = "X", 1 = "O", 0 = empty
        
    Returns:
        str: String representation of the tic-tac-toe board
    """
    # Convert vector values to board symbols
    symbols = []
    for val in vector:
        if val == -1:
            symbols.append("X")
        elif val == 1:
            symbols.append("O")
        else:
            symbols.append(" ")
    
    # Create the board layout
    board = f"""
 {symbols[0]} | {symbols[1]} | {symbols[2]} 
-----------
 {symbols[3]} | {symbols[4]} | {symbols[5]} 
-----------
 {symbols[6]} | {symbols[7]} | {symbols[8]} 
"""
    
    print(board)

if __name__ == "__main__":
    board_state = [1, 0, -1, 1, 1, 0, 0, 0, 0]
    comp_position = CompTurn(board_state)
    output = f"Place at Position {comp_position}"
    print(output)