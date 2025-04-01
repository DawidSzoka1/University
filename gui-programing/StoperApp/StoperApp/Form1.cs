using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace StoperApp
{
    public partial class Form1: Form
    {
        private Stopwatch stopwatch;
        private Timer timer;
        private int lapCount = 1;
        public Form1()
        {
            InitializeComponent();
            bStartStop.MouseDown += bStartStop_MouseDown;
            bResetLap.MouseDown += bResetLap_MouseDown;
          
            this.MaximizeBox = false;
            this.MinimizeBox = false;
            this.TopMost = true;
            this.Opacity = 1;
            this.Size = new System.Drawing.Size(170, 180); 
            this.StartPosition = FormStartPosition.Manual;
            this.Location = new System.Drawing.Point(Screen.PrimaryScreen.WorkingArea.Width - this.Width, 0);
            stopwatch = new Stopwatch();
            timer = new Timer();
            timer.Interval = 10;
            timer.Tick += Timer_Tick;
        }
        private void Timer_Tick(object sender, EventArgs e)
        {
            labelTime.Text = stopwatch.Elapsed.ToString(@"hh\:mm\:ss\.ff");
        }

        private void bStartStop_Click(object sender, EventArgs e){}

        private void bStartStop_MouseDown(object sender, MouseEventArgs e)
        {
            if (!stopwatch.IsRunning)
            {
                stopwatch.Start();
                timer.Start();
                bStartStop.Text = "Stop";
                bResetLap.Text = "Lap";

            }
            else
            {
                stopwatch.Stop();
                timer.Stop();
                bStartStop.Text = "Start";
                bResetLap.Text = "Reset";

            }
        }
        private void button2_Click(object sender, EventArgs e)
        {
       
        }
        private void bResetLap_MouseDown(object sender, MouseEventArgs e)
        {
            if (stopwatch.IsRunning)
            {
                listBoxLaps.Items.Add($"{lapCount++}: {stopwatch.Elapsed.ToString(@"hh\:mm\:ss\.ff")}");

            }
            else
            {
                stopwatch.Reset();
                labelTime.Text = "00:00:00.00";
                listBoxLaps.Items.Clear();
                lapCount = 1;
            }
        }
        private void Form1_Activated(object sender, EventArgs e)
        {
            this.Opacity = 1;
        }

        private void Form1_Deactivate(object sender, EventArgs e)
        {
            this.Opacity = 0.5;
        }
    }
}
