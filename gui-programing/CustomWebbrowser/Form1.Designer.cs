namespace CustomWebbrowser
{
    partial class Form1
    {
        /// <summary>
        /// Wymagana zmienna projektanta.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Wyczyść wszystkie używane zasoby.
        /// </summary>
        /// <param name="disposing">prawda, jeżeli zarządzane zasoby powinny zostać zlikwidowane; Fałsz w przeciwnym wypadku.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Kod generowany przez Projektanta formularzy systemu Windows

        /// <summary>
        /// Metoda wymagana do obsługi projektanta — nie należy modyfikować
        /// jej zawartości w edytorze kodu.
        /// </summary>
        private void InitializeComponent()
        {
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(Form1));
            this.toolStripContainer1 = new System.Windows.Forms.ToolStripContainer();
            this.splitContainer1 = new System.Windows.Forms.SplitContainer();
            this.webBrowser1 = new System.Windows.Forms.WebBrowser();
            this.richTextBoxHTML = new System.Windows.Forms.RichTextBox();
            this.menuStrip1 = new System.Windows.Forms.MenuStrip();
            this.fileToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.exitToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.helpToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.aboutToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.toolStrip1 = new System.Windows.Forms.ToolStrip();
            this.GoBackTS = new System.Windows.Forms.ToolStripButton();
            this.ForwardTS = new System.Windows.Forms.ToolStripButton();
            this.HomeTS = new System.Windows.Forms.ToolStripButton();
            this.RefreshTS = new System.Windows.Forms.ToolStripButton();
            this.UrlComboBox = new System.Windows.Forms.ToolStripComboBox();
            this.GoTS = new System.Windows.Forms.ToolStripButton();
            this.statusStrip1 = new System.Windows.Forms.StatusStrip();
            this.statusLabel = new System.Windows.Forms.ToolStripStatusLabel();
            this.toolStripContainer1.BottomToolStripPanel.SuspendLayout();
            this.toolStripContainer1.ContentPanel.SuspendLayout();
            this.toolStripContainer1.TopToolStripPanel.SuspendLayout();
            this.toolStripContainer1.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.splitContainer1)).BeginInit();
            this.splitContainer1.Panel1.SuspendLayout();
            this.splitContainer1.Panel2.SuspendLayout();
            this.splitContainer1.SuspendLayout();
            this.menuStrip1.SuspendLayout();
            this.toolStrip1.SuspendLayout();
            this.statusStrip1.SuspendLayout();
            this.SuspendLayout();
            // 
            // toolStripContainer1
            // 
            // 
            // toolStripContainer1.BottomToolStripPanel
            // 
            this.toolStripContainer1.BottomToolStripPanel.Controls.Add(this.statusStrip1);
            // 
            // toolStripContainer1.ContentPanel
            // 
            this.toolStripContainer1.ContentPanel.Controls.Add(this.splitContainer1);
            this.toolStripContainer1.ContentPanel.Size = new System.Drawing.Size(802, 373);
            this.toolStripContainer1.Location = new System.Drawing.Point(1, 1);
            this.toolStripContainer1.Name = "toolStripContainer1";
            this.toolStripContainer1.Size = new System.Drawing.Size(802, 449);
            this.toolStripContainer1.TabIndex = 0;
            this.toolStripContainer1.Text = "toolStripContainer1";
            // 
            // toolStripContainer1.TopToolStripPanel
            // 
            this.toolStripContainer1.TopToolStripPanel.Controls.Add(this.menuStrip1);
            this.toolStripContainer1.TopToolStripPanel.Controls.Add(this.toolStrip1);
            this.toolStripContainer1.TopToolStripPanel.Click += new System.EventHandler(this.toolStripContainer1_TopToolStripPanel_Click);
            // 
            // splitContainer1
            // 
            this.splitContainer1.Dock = System.Windows.Forms.DockStyle.Fill;
            this.splitContainer1.Location = new System.Drawing.Point(0, 0);
            this.splitContainer1.Name = "splitContainer1";
            // 
            // splitContainer1.Panel1
            // 
            this.splitContainer1.Panel1.Controls.Add(this.webBrowser1);
            // 
            // splitContainer1.Panel2
            // 
            this.splitContainer1.Panel2.Controls.Add(this.richTextBoxHTML);
            this.splitContainer1.Size = new System.Drawing.Size(802, 373);
            this.splitContainer1.SplitterDistance = 569;
            this.splitContainer1.TabIndex = 0;
            // 
            // webBrowser1
            // 
            this.webBrowser1.Dock = System.Windows.Forms.DockStyle.Fill;
            this.webBrowser1.Location = new System.Drawing.Point(0, 0);
            this.webBrowser1.MinimumSize = new System.Drawing.Size(20, 20);
            this.webBrowser1.Name = "webBrowser1";
            this.webBrowser1.Size = new System.Drawing.Size(569, 373);
            this.webBrowser1.TabIndex = 0;
            this.webBrowser1.DocumentCompleted += new System.Windows.Forms.WebBrowserDocumentCompletedEventHandler(this.webBrowser1_DocumentCompleted);
            // 
            // richTextBoxHTML
            // 
            this.richTextBoxHTML.Dock = System.Windows.Forms.DockStyle.Fill;
            this.richTextBoxHTML.Location = new System.Drawing.Point(0, 0);
            this.richTextBoxHTML.Name = "richTextBoxHTML";
            this.richTextBoxHTML.Size = new System.Drawing.Size(229, 373);
            this.richTextBoxHTML.TabIndex = 0;
            this.richTextBoxHTML.Text = "";
            // 
            // menuStrip1
            // 
            this.menuStrip1.Dock = System.Windows.Forms.DockStyle.None;
            this.menuStrip1.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.fileToolStripMenuItem});
            this.menuStrip1.Location = new System.Drawing.Point(0, 0);
            this.menuStrip1.Name = "menuStrip1";
            this.menuStrip1.Size = new System.Drawing.Size(802, 27);
            this.menuStrip1.TabIndex = 0;
            this.menuStrip1.Text = "menuStrip1";
            // 
            // fileToolStripMenuItem
            // 
            this.fileToolStripMenuItem.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.exitToolStripMenuItem,
            this.helpToolStripMenuItem});
            this.fileToolStripMenuItem.Name = "fileToolStripMenuItem";
            this.fileToolStripMenuItem.Size = new System.Drawing.Size(41, 23);
            this.fileToolStripMenuItem.Text = "File";
            // 
            // exitToolStripMenuItem
            // 
            this.exitToolStripMenuItem.Name = "exitToolStripMenuItem";
            this.exitToolStripMenuItem.Size = new System.Drawing.Size(180, 24);
            this.exitToolStripMenuItem.Text = "Exit";
            // 
            // helpToolStripMenuItem
            // 
            this.helpToolStripMenuItem.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.aboutToolStripMenuItem});
            this.helpToolStripMenuItem.Name = "helpToolStripMenuItem";
            this.helpToolStripMenuItem.Size = new System.Drawing.Size(180, 24);
            this.helpToolStripMenuItem.Text = "Help";
            // 
            // aboutToolStripMenuItem
            // 
            this.aboutToolStripMenuItem.Name = "aboutToolStripMenuItem";
            this.aboutToolStripMenuItem.Size = new System.Drawing.Size(180, 24);
            this.aboutToolStripMenuItem.Text = "About";
            // 
            // toolStrip1
            // 
            this.toolStrip1.Dock = System.Windows.Forms.DockStyle.None;
            this.toolStrip1.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.GoBackTS,
            this.ForwardTS,
            this.HomeTS,
            this.RefreshTS,
            this.UrlComboBox,
            this.GoTS});
            this.toolStrip1.Location = new System.Drawing.Point(3, 27);
            this.toolStrip1.Name = "toolStrip1";
            this.toolStrip1.Size = new System.Drawing.Size(396, 27);
            this.toolStrip1.TabIndex = 1;
            // 
            // GoBackTS
            // 
            this.GoBackTS.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Text;
            this.GoBackTS.Image = ((System.Drawing.Image)(resources.GetObject("GoBackTS.Image")));
            this.GoBackTS.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.GoBackTS.Name = "GoBackTS";
            this.GoBackTS.Size = new System.Drawing.Size(59, 24);
            this.GoBackTS.Text = "GoBack";
            this.GoBackTS.Click += new System.EventHandler(this.GoBackTS_Click);
            // 
            // ForwardTS
            // 
            this.ForwardTS.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Text;
            this.ForwardTS.Image = ((System.Drawing.Image)(resources.GetObject("ForwardTS.Image")));
            this.ForwardTS.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.ForwardTS.Name = "ForwardTS";
            this.ForwardTS.Size = new System.Drawing.Size(63, 24);
            this.ForwardTS.Text = "Forward";
            this.ForwardTS.Click += new System.EventHandler(this.ForwardTS_Click);
            // 
            // HomeTS
            // 
            this.HomeTS.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Text;
            this.HomeTS.Image = ((System.Drawing.Image)(resources.GetObject("HomeTS.Image")));
            this.HomeTS.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.HomeTS.Name = "HomeTS";
            this.HomeTS.Size = new System.Drawing.Size(50, 24);
            this.HomeTS.Text = "Home";
            this.HomeTS.Click += new System.EventHandler(this.HomeTS_Click);
            // 
            // RefreshTS
            // 
            this.RefreshTS.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Text;
            this.RefreshTS.Image = ((System.Drawing.Image)(resources.GetObject("RefreshTS.Image")));
            this.RefreshTS.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.RefreshTS.Name = "RefreshTS";
            this.RefreshTS.Size = new System.Drawing.Size(58, 24);
            this.RefreshTS.Text = "Refresh";
            this.RefreshTS.Click += new System.EventHandler(this.RefreshTS_Click);
            // 
            // UrlComboBox
            // 
            this.UrlComboBox.Name = "UrlComboBox";
            this.UrlComboBox.Size = new System.Drawing.Size(121, 27);
            // 
            // GoTS
            // 
            this.GoTS.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Text;
            this.GoTS.Image = ((System.Drawing.Image)(resources.GetObject("GoTS.Image")));
            this.GoTS.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.GoTS.Name = "GoTS";
            this.GoTS.Size = new System.Drawing.Size(31, 24);
            this.GoTS.Text = "Go";
            this.GoTS.Click += new System.EventHandler(this.GoTS_Click);
            // 
            // statusStrip1
            // 
            this.statusStrip1.Dock = System.Windows.Forms.DockStyle.None;
            this.statusStrip1.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.statusLabel});
            this.statusStrip1.Location = new System.Drawing.Point(0, 0);
            this.statusStrip1.Name = "statusStrip1";
            this.statusStrip1.Size = new System.Drawing.Size(802, 22);
            this.statusStrip1.TabIndex = 0;
            // 
            // statusLabel
            // 
            this.statusLabel.Name = "statusLabel";
            this.statusLabel.Size = new System.Drawing.Size(0, 17);
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(800, 450);
            this.Controls.Add(this.toolStripContainer1);
            this.MainMenuStrip = this.menuStrip1;
            this.Name = "Form1";
            this.Text = "Form1";
            this.toolStripContainer1.BottomToolStripPanel.ResumeLayout(false);
            this.toolStripContainer1.BottomToolStripPanel.PerformLayout();
            this.toolStripContainer1.ContentPanel.ResumeLayout(false);
            this.toolStripContainer1.TopToolStripPanel.ResumeLayout(false);
            this.toolStripContainer1.TopToolStripPanel.PerformLayout();
            this.toolStripContainer1.ResumeLayout(false);
            this.toolStripContainer1.PerformLayout();
            this.splitContainer1.Panel1.ResumeLayout(false);
            this.splitContainer1.Panel2.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.splitContainer1)).EndInit();
            this.splitContainer1.ResumeLayout(false);
            this.menuStrip1.ResumeLayout(false);
            this.menuStrip1.PerformLayout();
            this.toolStrip1.ResumeLayout(false);
            this.toolStrip1.PerformLayout();
            this.statusStrip1.ResumeLayout(false);
            this.statusStrip1.PerformLayout();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.ToolStripContainer toolStripContainer1;
        private System.Windows.Forms.SplitContainer splitContainer1;
        private System.Windows.Forms.WebBrowser webBrowser1;
        private System.Windows.Forms.RichTextBox richTextBoxHTML;
        private System.Windows.Forms.MenuStrip menuStrip1;
        private System.Windows.Forms.ToolStripMenuItem fileToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem exitToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem helpToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem aboutToolStripMenuItem;
        private System.Windows.Forms.ToolStrip toolStrip1;
        private System.Windows.Forms.ToolStripButton GoBackTS;
        private System.Windows.Forms.ToolStripButton ForwardTS;
        private System.Windows.Forms.ToolStripButton HomeTS;
        private System.Windows.Forms.ToolStripButton RefreshTS;
        private System.Windows.Forms.ToolStripComboBox UrlComboBox;
        private System.Windows.Forms.StatusStrip statusStrip1;
        private System.Windows.Forms.ToolStripStatusLabel statusLabel;
        private System.Windows.Forms.ToolStripButton GoTS;
    }
}

