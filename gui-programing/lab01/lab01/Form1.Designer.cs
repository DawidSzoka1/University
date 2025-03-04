namespace lab01
{
    partial class Form1
    {
        /// <summary>
        ///  Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        ///  Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        ///  Required method for Designer support - do not modify
        ///  the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            button1 = new Button();
            labe1 = new Label();
            label2 = new Label();
            label3 = new Label();
            tbA = new TextBox();
            tbB = new TextBox();
            tbWynik = new TextBox();
            groupBox1 = new GroupBox();
            rbAdd = new RadioButton();
            rbMinus = new RadioButton();
            rbMul = new RadioButton();
            rbDiv = new RadioButton();
            groupBox1.SuspendLayout();
            SuspendLayout();
            // 
            // button1
            // 
            button1.Location = new Point(187, 235);
            button1.Name = "button1";
            button1.Size = new Size(218, 72);
            button1.TabIndex = 0;
            button1.Text = "oblicz";
            button1.UseVisualStyleBackColor = true;
            button1.Click += button1_Click;
            // 
            // labe1
            // 
            labe1.AutoSize = true;
            labe1.Location = new Point(180, 84);
            labe1.Name = "labe1";
            labe1.Size = new Size(21, 19);
            labe1.TabIndex = 1;
            labe1.Text = "A:";
            labe1.Click += label1_Click;
            // 
            // label2
            // 
            label2.AutoSize = true;
            label2.Location = new Point(180, 121);
            label2.Name = "label2";
            label2.Size = new Size(20, 19);
            label2.TabIndex = 2;
            label2.Text = "B:";
            // 
            // label3
            // 
            label3.AutoSize = true;
            label3.Location = new Point(180, 158);
            label3.Name = "label3";
            label3.Size = new Size(44, 19);
            label3.TabIndex = 3;
            label3.Text = "wynik";
            // 
            // tbA
            // 
            tbA.Location = new Point(248, 74);
            tbA.Name = "tbA";
            tbA.Size = new Size(123, 26);
            tbA.TabIndex = 4;
            // 
            // tbB
            // 
            tbB.Location = new Point(248, 110);
            tbB.Name = "tbB";
            tbB.Size = new Size(126, 26);
            tbB.TabIndex = 5;
            tbB.TextChanged += tbB_TextChanged;
            // 
            // tbWynik
            // 
            tbWynik.Location = new Point(246, 154);
            tbWynik.Name = "tbWynik";
            tbWynik.ReadOnly = true;
            tbWynik.Size = new Size(134, 26);
            tbWynik.TabIndex = 6;
            // 
            // groupBox1
            // 
            groupBox1.Controls.Add(rbDiv);
            groupBox1.Controls.Add(rbMul);
            groupBox1.Controls.Add(rbMinus);
            groupBox1.Controls.Add(rbAdd);
            groupBox1.Location = new Point(507, 60);
            groupBox1.Name = "groupBox1";
            groupBox1.Size = new Size(198, 155);
            groupBox1.TabIndex = 7;
            groupBox1.TabStop = false;
            groupBox1.Text = "Dzialania";
            // 
            // rbAdd
            // 
            rbAdd.AutoSize = true;
            rbAdd.Checked = true;
            rbAdd.Location = new Point(11, 29);
            rbAdd.Name = "rbAdd";
            rbAdd.Size = new Size(37, 23);
            rbAdd.TabIndex = 0;
            rbAdd.TabStop = true;
            rbAdd.Text = "+";
            rbAdd.UseVisualStyleBackColor = true;
            // 
            // rbMinus
            // 
            rbMinus.AutoSize = true;
            rbMinus.Location = new Point(11, 63);
            rbMinus.Name = "rbMinus";
            rbMinus.Size = new Size(33, 23);
            rbMinus.TabIndex = 1;
            rbMinus.Text = "-";
            rbMinus.UseVisualStyleBackColor = true;
            // 
            // rbMul
            // 
            rbMul.AutoSize = true;
            rbMul.Location = new Point(10, 95);
            rbMul.Name = "rbMul";
            rbMul.Size = new Size(33, 23);
            rbMul.TabIndex = 2;
            rbMul.Text = "*";
            rbMul.UseVisualStyleBackColor = true;
            // 
            // rbDiv
            // 
            rbDiv.AutoSize = true;
            rbDiv.Location = new Point(10, 125);
            rbDiv.Name = "rbDiv";
            rbDiv.Size = new Size(32, 23);
            rbDiv.TabIndex = 3;
            rbDiv.Text = "/";
            rbDiv.UseVisualStyleBackColor = true;
            // 
            // Form1
            // 
            AutoScaleDimensions = new SizeF(8F, 19F);
            AutoScaleMode = AutoScaleMode.Font;
            ClientSize = new Size(800, 450);
            Controls.Add(groupBox1);
            Controls.Add(tbWynik);
            Controls.Add(tbB);
            Controls.Add(tbA);
            Controls.Add(label3);
            Controls.Add(label2);
            Controls.Add(labe1);
            Controls.Add(button1);
            Name = "Form1";
            Text = "Form1";
            Load += Form1_Load;
            groupBox1.ResumeLayout(false);
            groupBox1.PerformLayout();
            ResumeLayout(false);
            PerformLayout();
        }

        #endregion

        private Button button1;
        private Label labe1;
        private Label label2;
        private Label label3;
        private TextBox tbA;
        private TextBox tbB;
        private TextBox tbWynik;
        private GroupBox groupBox1;
        private RadioButton rbDiv;
        private RadioButton rbMul;
        private RadioButton rbMinus;
        private RadioButton rbAdd;
    }
}
