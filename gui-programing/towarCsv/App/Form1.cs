using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using static System.Windows.Forms.VisualStyles.VisualStyleElement.Button;
using static System.Windows.Forms.VisualStyles.VisualStyleElement;

namespace App
{
    public partial class Form1: Form
    {
        public Form1()
        {
            InitializeComponent();
            Count.ValueChanged += new EventHandler(Count_ValueChanged);
            PricePerUnit.ValueChanged += new EventHandler(Price_ValueChanged);
        }

        private void Count_ValueChanged(object sender, EventArgs e)
        {
            float count = (float)Count.Value;
            float pricePerUnit = (float)PricePerUnit.Value;
            tbAmount.Text = (count * pricePerUnit).ToString("0.00") + " zł";
        }

        private void Price_ValueChanged(object sender, EventArgs e)
        {
            float count = (float)Count.Value;
            float pricePerUnit = (float)PricePerUnit.Value;
            tbAmount.Text = (count * pricePerUnit).ToString("0.00") + " zł";
        }

        private void label3_Click(object sender, EventArgs e)
        {

        }

        private void label5_Click(object sender, EventArgs e)
        {

        }

        private void checkBox1_CheckedChanged(object sender, EventArgs e)
        {

        }

        private void Send_CheckedChanged(object sender, EventArgs e)
        {

        }

        private void label10_Click(object sender, EventArgs e)
        {

        }

        private void BPay_Click(object sender, EventArgs e)
        {
            if (Paid.Checked)
            {
                Paid.Checked = false;
            }
            else
            {
                Paid.Checked = true;
            }
    
        }

        private void BSend_Click(object sender, EventArgs e)
        {
            if (Send.Checked)
            {
                Send.Checked = false;
                DateOfSend.Visible = false;
            }
            else
            {
                Send.Checked = true;
                DateOfSend.Value = DateTime.Now;
                DateOfSend.Visible = true;
            }
            
        }

        private void BDelivered_Click(object sender, EventArgs e)
        {
            if (Delivered.Checked) {
                Delivered.Checked = false;
            }
            else
            {
                Delivered.Checked = true;
               
            }
               
        }

        private void label123_Click(object sender, EventArgs e)
        {

        }

        private void nowyToolStripMenuItem_Click(object sender, EventArgs e)
        {
            DialogResult result = MessageBox.Show("Czy na pewno chcesz usunąć wszystkie dane?",
                                          "Potwierdzenie",
                                          MessageBoxButtons.YesNo,
                                          MessageBoxIcon.Warning);
            
            if(result == DialogResult.Yes)
            {
                ProdName.Text = "";
                Paid.Checked = false;
                DateOfSend.Visible = false;
                Send.Checked = false;
                Delivered.Checked = false;
                Count.Value = 0;
                PricePerUnit.Value = 0;
                Producent.SelectedIndex = -1;
                tbDescription.Clear();
            }
        }

        private void zakończToolStripMenuItem_Click(object sender, EventArgs e)
        {
            Application.Exit();
        }

        private void otwórzToolStripMenuItem_Click(object sender, EventArgs e)
        {
            OpenFileDialog openFileDialog = new OpenFileDialog
            {
                Filter = "Pliki CSV (*.csv)|*.csv|Wszystkie pliki (*.*)|*.*"
            };

            if (openFileDialog.ShowDialog() == DialogResult.OK)
            {
                MessageBox.Show("Otworzono plik: " + openFileDialog.FileName);
                LoadFromCsv(openFileDialog.FileName);
            }
        }

        private void SaveToCsv()
        {
            try
            {
                using (SaveFileDialog saveFileDialog = new SaveFileDialog())
                {
                    saveFileDialog.Filter = "CSV files (*.csv)|*.csv";
                    if (saveFileDialog.ShowDialog() == DialogResult.OK)
                    {
                        StringBuilder sb = new StringBuilder();
                        sb.AppendLine($"{ProdName.Text};{Count.Text};{PricePerUnit.Text};{tbAmount.Text};" +
                            $"{Producent.SelectedItem};{Paid.Checked};{Send.Checked};" +
                            $"{DateOfSend.Value.ToShortDateString()};{Delivered.Checked};{tbDescription.Text}");
                        File.WriteAllText(saveFileDialog.FileName, sb.ToString());
                        MessageBox.Show("Dane zapisane");
                    }
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show("Zly zapis: " + ex.Message);
            }
        }
        private void LoadFromCsv(string filePath)
        {
            try
            {
                string[] lines = File.ReadAllLines(filePath);
                if (lines.Length > 0)
                {
                    string[] values = lines[0].Split(';');
                    ProdName.Text = values[0];
                    Count.Value =   decimal.Parse(values[1]);
                    PricePerUnit.Value = decimal.Parse(values[2]);
                    Producent.SelectedItem = values[4];
                    Paid.Checked = bool.Parse(values[5]);
                    Send.Checked = bool.Parse(values[6]);
                    DateOfSend.Value = DateTime.Parse(values[7]);
                    Delivered.Checked = bool.Parse(values[8]);
                    tbDescription.Text = values[9].Replace("\n", Environment.NewLine);

                }
            }
            catch (Exception ex)
            {
                MessageBox.Show("Błąd podczas wczytywania pliku CSV: " + ex.Message, "Błąd", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

        private void zapiszToolStripMenuItem_Click(object sender, EventArgs e)
        {
            SaveToCsv();
        }

        private void zapiszjakoToolStripMenuItem_Click(object sender, EventArgs e)
        {
            SaveToCsv();
        }

        private void tsbSave_Click(object sender, EventArgs e)
        {
            SaveToCsv();
        }

        private void tsbLoad_Click(object sender, EventArgs e)
        {
            OpenFileDialog openFileDialog = new OpenFileDialog
            {
                Filter = "Pliki CSV (*.csv)|*.csv|Wszystkie pliki (*.*)|*.*"
            };

            if (openFileDialog.ShowDialog() == DialogResult.OK)
            {
                MessageBox.Show("Otworzono plik: " + openFileDialog.FileName);
                LoadFromCsv(openFileDialog.FileName);
            }
        }

        private void plikToolStripMenuItem_Click(object sender, EventArgs e)
        {

        }
    }
}
