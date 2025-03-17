import random
import itertools
from collections import defaultdict

# letter frequency for balanced letter distribution
LETTER_FREQUENCIES = {
    'E': 12, 'T': 9, 'A': 8, 'O': 8, 'I': 7, 'N': 7, 'S': 7, 'H': 6, 'R': 6,
    'D': 4, 'L': 4, 'C': 3, 'U': 3, 'M': 3, 'W': 2, 'F': 2, 'G': 2, 'Y': 2,
    'P': 2, 'B': 1, 'V': 1, 'K': 1, 'J': 1, 'X': 1, 'Q': 1, 'Z': 1
}
LETTERS = list(itertools.chain(*[[k] * v for k, v in LETTER_FREQUENCIES.items()]))

# Generates a completely random board based on letter frequencies
def generate_random_board():
    return ''.join(random.choices(LETTERS, k=16))  # Create 4x4 board

class TrieNode:
    def __init__(self):
        self.children = defaultdict(TrieNode)
        self.is_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for letter in word:
            node = node.children[letter]
        node.is_word = True

    def search(self, word):
        node = self.root
        for letter in word:
            if letter not in node.children:
                return False
            node = node.children[letter]
        return node.is_word

def load_dictionary():
    trie = Trie()
    with open("words.txt") as f: 
        for word in f.read().split():
            if len(word) >= 3:  # Only use words of 3+ letters
                trie.insert(word.upper())
    return trie



MYTRIE = load_dictionary()
def find_words(board):
    N = 4
    board = [list(board[i * N:(i + 1) * N]) for i in range(N)]
    found_words = set()
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    def dfs(x, y, node, word, visited):
        if node.is_word:
            found_words.add(word)

        visited.add((x, y))
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < N and 0 <= ny < N and (nx, ny) not in visited:
                letter = board[nx][ny]
                if letter in node.children:
                    dfs(nx, ny, node.children[letter], word + letter, visited.copy())

    for i in range(N):
        for j in range(N):
            letter = board[i][j]
            if letter in MYTRIE.root.children:
                dfs(i, j, MYTRIE.root.children[letter], letter, set())

    return found_words

def get_trie():
    return MYTRIE






