using System;
using System.Diagnostics;
using System.Drawing;
using System.Windows.Forms;

public class StoperForm : Form
{
    private Label timeLabel;
    private Button startStopButton;
    private Timer timer;
    private Stopwatch stopwatch;

    public StoperForm()
    {
        // Ustawienia formularza
        this.Text = "Stoper";
        this.Size = new Size(200, 100);
        this.FormBorderStyle = FormBorderStyle.FixedSingle;
        this.MaximizeBox = false;
        this.MinimizeBox = false;
        this.TopMost = true;
        this.Opacity = 1.0;

        // Inicjalizacja stopera i timera
        stopwatch = new Stopwatch();
        timer = new Timer { Interval = 50 }; // 50ms dla przyzwoitej responsywności
        timer.Tick += Timer_Tick;

        // Etykieta wyświetlająca czas
        timeLabel = new Label
        {
            Text = "00:00.00",
            AutoSize = true,
            Font = new Font("Arial", 14, FontStyle.Bold),
            Location = new Point(10, 10)
        };

        // Przycisk start/stop
        startStopButton = new Button
        {
            Text = "Start",
            Size = new Size(80, 30),
            Location = new Point(10, 40)
        };
        startStopButton.MouseDown += StartStopButton_MouseDown;

        // Dodanie elementów do formularza
        this.Controls.Add(timeLabel);
        this.Controls.Add(startStopButton);

        // Obsługa zdarzeń aktywacji/dezaktywacji
        this.Activated += (s, e) => this.Opacity = 1.0;
        this.Deactivate += (s, e) => this.Opacity = 0.5;
    }

    private void Timer_Tick(object sender, EventArgs e)
    {
        timeLabel.Text = stopwatch.Elapsed.ToString("mm\:ss\.ff");
    }

    private void StartStopButton_MouseDown(object sender, MouseEventArgs e)
    {
        if (stopwatch.IsRunning)
        {
            stopwatch.Stop();
            timer.Stop();
            startStopButton.Text = "Start";
        }
        else
        {
            stopwatch.Start();
            timer.Start();
            startStopButton.Text = "Stop";
        }
    }

    [STAThread]
    public static void Main()
    {
        Application.EnableVisualStyles();
        Application.SetCompatibleTextRenderingDefault(false);
        Application.Run(new StoperForm());
    }
}
