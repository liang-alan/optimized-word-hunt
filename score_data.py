import generate_data

WORD_SCORES = {
    3: 100,
    4: 400,
    5: 800,
    6: 1200,
    7: 1600,
    8: 2000,
    9: 2400
}

def score_board(board):
    if (len(board) != 16):
        print("ERROR: Board must be 4x4")
        return
    words = generate_data.find_words(board)
    total_score = 0
    for word in words:
        # default 0 if not in dictionary
        total_score += WORD_SCORES.get(len(word), 0)
    return total_score

board = "ASARNITTAPERREIR"
print(board, score_board(board))
