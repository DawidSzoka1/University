using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace AnimatedClock
{
    public partial class Form1: Form
    {
        public Form1()
        {
            InitializeComponent();
            timer1.Tick += timer1_Tick;
            timer1.Start();
        }

        private void timer1_Tick(object sender, EventArgs e)
        {
            pictureBox1.Invalidate();
        }
        private void pictureBox1_Paint(object sender, PaintEventArgs e)
        {
            Graphics g = e.Graphics;
            g.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.AntiAlias;

            int width = pictureBox1.Width;
            int height = pictureBox1.Height;
            Point center = new Point(width / 2, height / 2);
            int radius = Math.Min(width, height) / 2 - 10;

            // Rysuj tarczę zegara
            g.FillEllipse(Brushes.White, center.X - radius, center.Y - radius, radius * 2, radius * 2);
            g.DrawEllipse(Pens.Black, center.X - radius, center.Y - radius, radius * 2, radius * 2);

            // Rysuj cyfry (1-12)
            for (int i = 1; i <= 12; i++)
            {
                double angle = (i * 30 - 90) * Math.PI / 180;
                int tx = (int)(center.X + (radius - 25) * Math.Cos(angle));
                int ty = (int)(center.Y + (radius - 25) * Math.Sin(angle));
                string text = i.ToString();
                SizeF size = g.MeasureString(text, this.Font);
                g.DrawString(text, this.Font, Brushes.Black, tx - size.Width / 2, ty - size.Height / 2);
            }

            // Pobierz aktualny czas
            DateTime now = DateTime.Now;
            float secAngle = now.Second * 6;
            float minAngle = now.Minute * 6 + now.Second * 0.1f;
            float hourAngle = (now.Hour % 12) * 30 + now.Minute * 0.5f;

            // Wskazówki
            DrawHand(g, center, hourAngle, radius * 0.5f, 6, Brushes.Black); // godzinowa
            DrawHand(g, center, minAngle, radius * 0.7f, 4, Brushes.Blue);   // minutowa
            DrawHand(g, center, secAngle, radius * 0.9f, 2, Brushes.Red);    // sekundowa

            g.FillEllipse(Brushes.Black, center.X - 5, center.Y - 5, 10, 10);
        }

        private void DrawHand(Graphics g, Point center, float angleDeg, float length, float thickness, Brush brush)
        {
            double angleRad = (Math.PI / 180) * (angleDeg - 90);
            float x = (float)(center.X + length * Math.Cos(angleRad));
            float y = (float)(center.Y + length * Math.Sin(angleRad));
            Pen pen = new Pen(brush, thickness);
            pen.EndCap = System.Drawing.Drawing2D.LineCap.Round;
            g.DrawLine(pen, center.X, center.Y, x, y);
        }
        private void pictureBox1_Click(object sender, EventArgs e)
        {

        }
    }
}
