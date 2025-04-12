using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace TicTacToeApp
{
    public partial class Form1: Form
    {
        private GameTicTacToe game;
        private Button[,] buttons = new Button[3, 3];
        public Form1()
        {
            InitializeComponent();
            game = new GameTicTacToe();
            CreateGridButtons();
        }

        private void CreateGridButtons()
        {
            int size = 60;

            for (int x = 0; x < 3; x++)
            {
                for (int y = 0; y < 3; y++)
                {
                    Button btn = new Button();
                    btn.Width = size;
                    btn.Height = size;
                    btn.Location = new Point(x * size, y * size);
                    btn.Tag = new Point(x, y);
                    btn.Font = new Font(FontFamily.GenericSansSerif, 20, FontStyle.Bold);
                    btn.Click += Btn_Click;

                    panel1.Controls.Add(btn);
                    buttons[x, y] = btn;
                }
            }
        }

        private void Btn_Click(object sender, EventArgs e)
        {
            Button btn = (Button)sender;
            Point p = (Point)btn.Tag;

            int symbol = game.MakeMove(p.X, p.Y);

            if (symbol == 1) btn.Text = "X";
            else if (symbol == 4) btn.Text = "O";
            else return;

            labelLog.Text += $"{(symbol == 1 ? "X" : "O")} na ({p.X + 1},{p.Y + 1})\n";

            int result = game.CheckWinner();

            if (result == 1)
            {
                MessageBox.Show("Wygrał gracz X!");
                DisableButtons();
            }
            else if (result == 4)
            {
                MessageBox.Show("Wygrał gracz O!");
                DisableButtons();
            }
            else if (result == 0)
            {
                MessageBox.Show("Remis!");
            }
        }

        private void DisableButtons()
        {
            foreach (Button btn in buttons)
            {
                btn.Enabled = false;
            }
        }

        private void resteujToolStripMenuItem_Click(object sender, EventArgs e)
        {
            game.ResetBoard();
            foreach (Button b in buttons)
            {
                b.Text = "";
                b.Enabled = true;
            }
            labelLog.Text = "";
        }
    }
}
