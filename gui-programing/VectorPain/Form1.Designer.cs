namespace VectorPain
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
            this.panel1 = new System.Windows.Forms.Panel();
            this.panelShadow = new System.Windows.Forms.Panel();
            this.toolStrip1 = new System.Windows.Forms.ToolStrip();
            this.nowyToolStripButton = new System.Windows.Forms.ToolStripButton();
            this.otwórzToolStripButton = new System.Windows.Forms.ToolStripButton();
            this.zapiszToolStripButton = new System.Windows.Forms.ToolStripButton();
            this.drukujToolStripButton = new System.Windows.Forms.ToolStripButton();
            this.toolStripSeparator = new System.Windows.Forms.ToolStripSeparator();
            this.wytnijToolStripButton = new System.Windows.Forms.ToolStripButton();
            this.kopiujToolStripButton = new System.Windows.Forms.ToolStripButton();
            this.wklejToolStripButton = new System.Windows.Forms.ToolStripButton();
            this.toolStripSeparator1 = new System.Windows.Forms.ToolStripSeparator();
            this.pomocToolStripButton = new System.Windows.Forms.ToolStripButton();
            this.Point = new System.Windows.Forms.ToolStripButton();
            this.Line = new System.Windows.Forms.ToolStripButton();
            this.Rect = new System.Windows.Forms.ToolStripButton();
            this.Ellipse = new System.Windows.Forms.ToolStripButton();
            this.colorDialog1 = new System.Windows.Forms.ColorDialog();
            this.btnPenColor = new System.Windows.Forms.ToolStripButton();
            this.btnFillColor = new System.Windows.Forms.ToolStripButton();
            this.comboBox1 = new System.Windows.Forms.ToolStripComboBox();
            this.panel2 = new System.Windows.Forms.Panel();
            this.panel1.SuspendLayout();
            this.toolStrip1.SuspendLayout();
            this.SuspendLayout();
            // 
            // panel1
            // 
            this.panel1.Controls.Add(this.toolStrip1);
            this.panel1.Location = new System.Drawing.Point(12, 12);
            this.panel1.Name = "panel1";
            this.panel1.Size = new System.Drawing.Size(686, 400);
            this.panel1.TabIndex = 0;
            this.panel1.Paint += new System.Windows.Forms.PaintEventHandler(this.panel1_Paint);
            this.panel1.MouseDown += new System.Windows.Forms.MouseEventHandler(this.panel1_MouseDown);
            this.panel1.MouseMove += new System.Windows.Forms.MouseEventHandler(this.panel1_MouseMove);
            this.panel1.MouseUp += new System.Windows.Forms.MouseEventHandler(this.panel1_MouseUp);
            // 
            // panelShadow
            // 
            this.panelShadow.BackColor = System.Drawing.Color.LightGray;
            this.panelShadow.Location = new System.Drawing.Point(12, 410);
            this.panelShadow.Name = "panelShadow";
            this.panelShadow.Size = new System.Drawing.Size(693, 84);
            this.panelShadow.TabIndex = 1;
            // 
            // toolStrip1
            // 
            this.toolStrip1.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.nowyToolStripButton,
            this.otwórzToolStripButton,
            this.zapiszToolStripButton,
            this.drukujToolStripButton,
            this.toolStripSeparator,
            this.wytnijToolStripButton,
            this.kopiujToolStripButton,
            this.wklejToolStripButton,
            this.toolStripSeparator1,
            this.pomocToolStripButton,
            this.Point,
            this.Line,
            this.Rect,
            this.Ellipse,
            this.btnPenColor,
            this.btnFillColor,
            this.comboBox1});
            this.toolStrip1.Location = new System.Drawing.Point(0, 0);
            this.toolStrip1.Name = "toolStrip1";
            this.toolStrip1.Size = new System.Drawing.Size(686, 27);
            this.toolStrip1.TabIndex = 0;
            this.toolStrip1.Text = "toolStrip1";
            // 
            // nowyToolStripButton
            // 
            this.nowyToolStripButton.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.nowyToolStripButton.Image = ((System.Drawing.Image)(resources.GetObject("nowyToolStripButton.Image")));
            this.nowyToolStripButton.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.nowyToolStripButton.Name = "nowyToolStripButton";
            this.nowyToolStripButton.Size = new System.Drawing.Size(23, 24);
            this.nowyToolStripButton.Text = "&Nowy";
            this.nowyToolStripButton.Click += new System.EventHandler(this.nowyToolStripButton_Click);
            // 
            // otwórzToolStripButton
            // 
            this.otwórzToolStripButton.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.otwórzToolStripButton.Enabled = false;
            this.otwórzToolStripButton.Image = ((System.Drawing.Image)(resources.GetObject("otwórzToolStripButton.Image")));
            this.otwórzToolStripButton.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.otwórzToolStripButton.Name = "otwórzToolStripButton";
            this.otwórzToolStripButton.Size = new System.Drawing.Size(23, 24);
            this.otwórzToolStripButton.Text = "&Otwórz";
            // 
            // zapiszToolStripButton
            // 
            this.zapiszToolStripButton.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.zapiszToolStripButton.DoubleClickEnabled = true;
            this.zapiszToolStripButton.Enabled = false;
            this.zapiszToolStripButton.Image = ((System.Drawing.Image)(resources.GetObject("zapiszToolStripButton.Image")));
            this.zapiszToolStripButton.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.zapiszToolStripButton.Name = "zapiszToolStripButton";
            this.zapiszToolStripButton.Size = new System.Drawing.Size(23, 24);
            this.zapiszToolStripButton.Text = "&Zapisz";
            // 
            // drukujToolStripButton
            // 
            this.drukujToolStripButton.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.drukujToolStripButton.Enabled = false;
            this.drukujToolStripButton.Image = ((System.Drawing.Image)(resources.GetObject("drukujToolStripButton.Image")));
            this.drukujToolStripButton.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.drukujToolStripButton.Name = "drukujToolStripButton";
            this.drukujToolStripButton.Size = new System.Drawing.Size(23, 24);
            this.drukujToolStripButton.Text = "&Drukuj";
            // 
            // toolStripSeparator
            // 
            this.toolStripSeparator.Name = "toolStripSeparator";
            this.toolStripSeparator.Size = new System.Drawing.Size(6, 27);
            // 
            // wytnijToolStripButton
            // 
            this.wytnijToolStripButton.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.wytnijToolStripButton.Enabled = false;
            this.wytnijToolStripButton.Image = ((System.Drawing.Image)(resources.GetObject("wytnijToolStripButton.Image")));
            this.wytnijToolStripButton.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.wytnijToolStripButton.Name = "wytnijToolStripButton";
            this.wytnijToolStripButton.Size = new System.Drawing.Size(23, 24);
            this.wytnijToolStripButton.Text = "&Wytnij";
            // 
            // kopiujToolStripButton
            // 
            this.kopiujToolStripButton.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.kopiujToolStripButton.Enabled = false;
            this.kopiujToolStripButton.Image = ((System.Drawing.Image)(resources.GetObject("kopiujToolStripButton.Image")));
            this.kopiujToolStripButton.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.kopiujToolStripButton.Name = "kopiujToolStripButton";
            this.kopiujToolStripButton.Size = new System.Drawing.Size(23, 24);
            this.kopiujToolStripButton.Text = "&Kopiuj";
            // 
            // wklejToolStripButton
            // 
            this.wklejToolStripButton.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.wklejToolStripButton.Enabled = false;
            this.wklejToolStripButton.Image = ((System.Drawing.Image)(resources.GetObject("wklejToolStripButton.Image")));
            this.wklejToolStripButton.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.wklejToolStripButton.Name = "wklejToolStripButton";
            this.wklejToolStripButton.Size = new System.Drawing.Size(23, 24);
            this.wklejToolStripButton.Text = "&Wklej";
            // 
            // toolStripSeparator1
            // 
            this.toolStripSeparator1.Name = "toolStripSeparator1";
            this.toolStripSeparator1.Size = new System.Drawing.Size(6, 27);
            // 
            // pomocToolStripButton
            // 
            this.pomocToolStripButton.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.pomocToolStripButton.Enabled = false;
            this.pomocToolStripButton.Image = ((System.Drawing.Image)(resources.GetObject("pomocToolStripButton.Image")));
            this.pomocToolStripButton.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.pomocToolStripButton.Name = "pomocToolStripButton";
            this.pomocToolStripButton.Size = new System.Drawing.Size(23, 24);
            this.pomocToolStripButton.Text = "&Pomoc";
            // 
            // Point
            // 
            this.Point.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Text;
            this.Point.Image = ((System.Drawing.Image)(resources.GetObject("Point.Image")));
            this.Point.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.Point.Name = "Point";
            this.Point.Size = new System.Drawing.Size(44, 24);
            this.Point.Text = "Point";
            this.Point.Click += new System.EventHandler(this.Point_Click);
            // 
            // Line
            // 
            this.Line.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Text;
            this.Line.Image = ((System.Drawing.Image)(resources.GetObject("Line.Image")));
            this.Line.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.Line.Name = "Line";
            this.Line.Size = new System.Drawing.Size(38, 24);
            this.Line.Text = "Line";
            this.Line.Click += new System.EventHandler(this.Line_Click);
            // 
            // Rect
            // 
            this.Rect.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Text;
            this.Rect.Image = ((System.Drawing.Image)(resources.GetObject("Rect.Image")));
            this.Rect.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.Rect.Name = "Rect";
            this.Rect.Size = new System.Drawing.Size(39, 24);
            this.Rect.Text = "Rect";
            this.Rect.Click += new System.EventHandler(this.Rect_Click);
            // 
            // Ellipse
            // 
            this.Ellipse.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Text;
            this.Ellipse.Image = ((System.Drawing.Image)(resources.GetObject("Ellipse.Image")));
            this.Ellipse.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.Ellipse.Name = "Ellipse";
            this.Ellipse.Size = new System.Drawing.Size(50, 24);
            this.Ellipse.Text = "Ellipse";
            this.Ellipse.Click += new System.EventHandler(this.Ellipse_Click);
            // 
            // btnPenColor
            // 
            this.btnPenColor.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Text;
            this.btnPenColor.Image = ((System.Drawing.Image)(resources.GetObject("btnPenColor.Image")));
            this.btnPenColor.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.btnPenColor.Name = "btnPenColor";
            this.btnPenColor.Size = new System.Drawing.Size(45, 24);
            this.btnPenColor.Text = "Pióro";
            this.btnPenColor.Click += new System.EventHandler(this.btnPenColor_Click);
            // 
            // btnFillColor
            // 
            this.btnFillColor.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Text;
            this.btnFillColor.Image = ((System.Drawing.Image)(resources.GetObject("btnFillColor.Image")));
            this.btnFillColor.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.btnFillColor.Name = "btnFillColor";
            this.btnFillColor.Size = new System.Drawing.Size(88, 24);
            this.btnFillColor.Text = "Wypełnienie";
            this.btnFillColor.Click += new System.EventHandler(this.btnFillColor_Click);
            // 
            // comboBox1
            // 
            this.comboBox1.Items.AddRange(new object[] {
            "1",
            "2",
            "4",
            "6",
            "8"});
            this.comboBox1.Name = "comboBox1";
            this.comboBox1.Size = new System.Drawing.Size(121, 27);
            this.comboBox1.TextChanged += new System.EventHandler(this.comboBox1_TextChanged);
            // 
            // panel2
            // 
            this.panel2.BackColor = System.Drawing.Color.LightGray;
            this.panel2.Location = new System.Drawing.Point(694, 12);
            this.panel2.Name = "panel2";
            this.panel2.Size = new System.Drawing.Size(156, 482);
            this.panel2.TabIndex = 2;
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(1006, 603);
            this.Controls.Add(this.panel2);
            this.Controls.Add(this.panelShadow);
            this.Controls.Add(this.panel1);
            this.Name = "Form1";
            this.Text = "Form1";
            this.panel1.ResumeLayout(false);
            this.panel1.PerformLayout();
            this.toolStrip1.ResumeLayout(false);
            this.toolStrip1.PerformLayout();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.Panel panel1;
        private System.Windows.Forms.Panel panelShadow;
        private System.Windows.Forms.ToolStrip toolStrip1;
        private System.Windows.Forms.ToolStripButton nowyToolStripButton;
        private System.Windows.Forms.ToolStripButton otwórzToolStripButton;
        private System.Windows.Forms.ToolStripButton zapiszToolStripButton;
        private System.Windows.Forms.ToolStripButton drukujToolStripButton;
        private System.Windows.Forms.ToolStripSeparator toolStripSeparator;
        private System.Windows.Forms.ToolStripButton wytnijToolStripButton;
        private System.Windows.Forms.ToolStripButton kopiujToolStripButton;
        private System.Windows.Forms.ToolStripButton wklejToolStripButton;
        private System.Windows.Forms.ToolStripSeparator toolStripSeparator1;
        private System.Windows.Forms.ToolStripButton pomocToolStripButton;
        private System.Windows.Forms.ToolStripButton Point;
        private System.Windows.Forms.ToolStripButton Line;
        private System.Windows.Forms.ToolStripButton Rect;
        private System.Windows.Forms.ToolStripButton Ellipse;
        private System.Windows.Forms.ColorDialog colorDialog1;
        private System.Windows.Forms.ToolStripButton btnPenColor;
        private System.Windows.Forms.ToolStripButton btnFillColor;
        private System.Windows.Forms.ToolStripComboBox comboBox1;
        private System.Windows.Forms.Panel panel2;
    }
}

