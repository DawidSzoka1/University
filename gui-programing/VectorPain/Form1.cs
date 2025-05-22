using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using static System.Windows.Forms.LinkLabel;

namespace VectorPain
{
    public partial class Form1 : Form
    {
        List<Point> pp = new List<Point>();
        List<Rectangle> lines = new List<Rectangle>();
        List<Rectangle> rectangles = new List<Rectangle>();
        List<Rectangle> ellipses = new List<Rectangle>();
        Point start, temp;
        bool drawing = false;
        Pen currentPen = new Pen(Color.Black, 1);
        Brush currentBrush = new SolidBrush(Color.White);
        enum Tool { Point, Line, Rect, Ellipse }
        Tool currentTool = Tool.Point;

        public Form1()
        {
            InitializeComponent();
        }

        private void btnPenColor_Click(object sender, EventArgs e)
        {
            if (colorDialog1.ShowDialog() == DialogResult.OK)
                currentPen = new Pen(colorDialog1.Color, currentPen.Width);
        }

        private void btnFillColor_Click(object sender, EventArgs e)
        {
            if (colorDialog1.ShowDialog() == DialogResult.OK)
                currentBrush = new SolidBrush(colorDialog1.Color);
        }

        private void comboBox1_TextChanged(object sender, EventArgs e)
        {
            if (int.TryParse(comboBox1.Text, out int width) && width > 0)
            {
            
                currentPen = new Pen(currentPen.Color, width);
                panel1.Invalidate();
            }
        }

        private void btnPenColor_Paint(object sender, PaintEventArgs e)
        {
            Rectangle r = new Rectangle(2, 2, btnPenColor.Width - 4, btnPenColor.Height - 4);
            e.Graphics.FillRectangle(new SolidBrush(currentPen.Color), r);
            e.Graphics.DrawRectangle(Pens.Black, r);
            if (colorDialog1.ShowDialog() == DialogResult.OK)
            {
                // Używamy aktualnej grubości i nowego koloru
                currentPen = new Pen(colorDialog1.Color, currentPen.Width);
                panel1.Invalidate();
            }
        }

        private void panel1_MouseDown(object sender, MouseEventArgs e)
        {
            pp.Add(new Point(e.X, e.Y));
            panel1.Invalidate();
            if (currentTool == Tool.Point)
            {
                pp.Add(start);
                panel1.Invalidate();
            }
        }
        private void panel1_MouseUp(object sender, MouseEventArgs e)
        {
            drawing = false;

            Point end = new Point(e.X, e.Y);
            Rectangle r = GetRectangleFromPoints(start, end);

            switch (currentTool)
            {
                case Tool.Line:
                    lines.Add(new Rectangle(start.X, start.Y, end.X, end.Y)); // linia: przechowujemy dwa punkty jako Rectangle
                    break;
                case Tool.Rect:
                    rectangles.Add(r);
                    break;
                case Tool.Ellipse:
                    ellipses.Add(r);
                    break;
            }

            panel1.Invalidate();
        }
        private void panel1_MouseMove(object sender, MouseEventArgs e)
        {
            if (drawing && currentTool != Tool.Point)
            {
                temp = new Point(e.X, e.Y);
                panel1.Invalidate(); // podgląd
            }
        }

        private void panel1_Paint(object sender, PaintEventArgs e)
        {
            foreach (Point p in pp)
                e.Graphics.FillRectangle(currentPen.Brush, p.X, p.Y, 2, 2);

            // Linie
            foreach (Rectangle r in lines)
                e.Graphics.DrawLine(currentPen, r.Left, r.Top, r.Right, r.Bottom);

            // Prostokąty
            foreach (Rectangle r in rectangles)
            {
                e.Graphics.FillRectangle(currentBrush, r);       // wypełnienie
                e.Graphics.DrawRectangle(currentPen, r);         // obrys
            }

            // Elipsy
            foreach (Rectangle r in ellipses)
            {
                e.Graphics.FillEllipse(currentBrush, r);
                e.Graphics.DrawEllipse(currentPen, r);
            }

            // Podgląd figury podczas rysowania
            if (drawing && currentTool != Tool.Point)
            {
                Rectangle r = GetRectangleFromPoints(start, temp);

                switch (currentTool)
                {
                    case Tool.Line:
                        e.Graphics.DrawLine(currentPen, start, temp);
                        break;
                    case Tool.Rect:
                        e.Graphics.FillRectangle(currentBrush, r);
                        e.Graphics.DrawRectangle(currentPen, r);
                        break;
                    case Tool.Ellipse:
                        e.Graphics.FillEllipse(currentBrush, r);
                        e.Graphics.DrawEllipse(currentPen, r);
                        break;
                }
            }
        }
        private Rectangle GetRectangleFromPoints(Point p1, Point p2)
        {
            return new Rectangle(
                Math.Min(p1.X, p2.X),
                Math.Min(p1.Y, p2.Y),
                Math.Abs(p1.X - p2.X),
                Math.Abs(p1.Y - p2.Y));
        }

        private void nowyToolStripButton_Click(object sender, EventArgs e)
        {
            pp.Clear();
            lines.Clear();
            rectangles.Clear();
            ellipses.Clear();
            panel1.Invalidate();
        }

        private void Point_Click(object sender, EventArgs e)
        {
            currentTool = Tool.Point;
        }

        private void Line_Click(object sender, EventArgs e)
        {
            currentTool = Tool.Line;
        }

        private void Rect_Click(object sender, EventArgs e)
        {
            currentTool = Tool.Rect;
        }

        private void Ellipse_Click(object sender, EventArgs e)
        {
            currentTool = Tool.Ellipse;
        }
    }
}
