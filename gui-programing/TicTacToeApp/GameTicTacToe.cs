using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TicTacToeApp
{
    public class GameTicTacToe
    {
        private int[,] board = new int[3, 3];
        private int moveCount = 0;
        private int nextMove = 1; // 1 = X, 4 = O

        public int MakeMove(int x, int y)
        {
            if (board[x, y] == 0)
            {
                board[x, y] = nextMove;
                moveCount++;
                int current = nextMove;
                nextMove = 5 - nextMove; // switch between 1 and 4
                return current;
            }
            return 0;
        }

        public void ResetBoard()
        {
            board = new int[3, 3];
            moveCount = 0;
            nextMove = 1;
        }

        public int CheckWinner()
        {
            for (int i = 0; i < 3; i++)
            {
                int row = board[i, 0] + board[i, 1] + board[i, 2];
                int col = board[0, i] + board[1, i] + board[2, i];
                if (row == 3 || col == 3) return 1; // X wins
                if (row == 12 || col == 12) return 4; // O wins
            }

            int diag1 = board[0, 0] + board[1, 1] + board[2, 2];
            int diag2 = board[0, 2] + board[1, 1] + board[2, 0];

            if (diag1 == 3 || diag2 == 3) return 1;
            if (diag1 == 12 || diag2 == 12) return 4;

            if (moveCount >= 9) return 0; // draw

            return -1; // no result yet
        }
    }
}
