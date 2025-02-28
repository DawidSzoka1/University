using System.Numerics;

namespace lab01
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }

        private void Form1_Load(object sender, EventArgs e)
        {

        }

        private void button1_Click(object sender, EventArgs e)
        {
            float x = 0, y = 0, result = 0;
            try
            {
                x = float.Parse(tbA.Text);
                tbA.BackColor = Color.White;
            }
            catch (Exception)
            {
                tbA.BackColor = Color.Red;
              
            }
            try
            {
                y = float.Parse(tbB.Text);
                tbB.BackColor = Color.White;
            }
            catch (Exception)
            {
                tbB.BackColor = Color.Red;
                
            }
            if (rbAdd.Checked)
            {
                result = x + y;
            }else if (rbMinus.Checked)
            {
                result = x - y;
            }else if (rbMul.Checked)
            {
                result = x * y;
            }else if (rbDiv.Checked)
            {
                result = x / y;
            }
                tbWynik.Text = result.ToString();
        }

        private void label1_Click(object sender, EventArgs e)
        {

        }

        private void tbB_TextChanged(object sender, EventArgs e)
        {

        }
    }
}
