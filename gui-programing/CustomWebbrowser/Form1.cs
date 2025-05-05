using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace CustomWebbrowser
{
    public partial class Form1: Form
    {
        public Form1()
        {
          
           InitializeComponent();
            webBrowser1.ScriptErrorsSuppressed = true;

        }
        private List<string> history = new List<string>();

        private void toolStripContainer1_TopToolStripPanel_Click(object sender, EventArgs e)
        {

        }

        private void GoTS_Click(object sender, EventArgs e)
        {
            string url = UrlComboBox.Text;
            if (!url.StartsWith("http")) url = "http://" + url;
            webBrowser1.Navigate(url);
        }

        private void webBrowser1_DocumentCompleted(object sender, WebBrowserDocumentCompletedEventArgs e)
        {
            string html = webBrowser1.DocumentText;
            richTextBoxHTML.Text = html;

            if (!history.Contains(webBrowser1.Url.ToString()))
            {
                history.Add(webBrowser1.Url.ToString());
                UrlComboBox.Items.Add(webBrowser1.Url.ToString());
            }

            statusLabel.Text = "Załadowano: " + webBrowser1.Url.ToString();
        }


        private void GoBackTS_Click(object sender, EventArgs e)
        {
            if (webBrowser1.CanGoBack) webBrowser1.GoBack();
        }

        private void ForwardTS_Click(object sender, EventArgs e)
        {
            if (webBrowser1.CanGoForward) webBrowser1.GoForward();
        }

        private void HomeTS_Click(object sender, EventArgs e)
        {
            webBrowser1.GoHome();
        }

        private void RefreshTS_Click(object sender, EventArgs e)
        {
            webBrowser1.Refresh();
        }
    }
}
