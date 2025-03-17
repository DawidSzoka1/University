namespace main
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
            this.components = new System.ComponentModel.Container();
            this.bStartStop = new System.Windows.Forms.Button();
            this.bResetLab = new System.Windows.Forms.Button();
            this.label1 = new System.Windows.Forms.Label();
            this.timer1 = new System.Windows.Forms.Timer(this.components);
            this.SuspendLayout();
            // 
            // bStartStop
            // 
            this.bStartStop.Location = new System.Drawing.Point(332, 24);
            this.bStartStop.Name = "bStartStop";
            this.bStartStop.Size = new System.Drawing.Size(122, 52);
            this.bStartStop.TabIndex = 0;
            this.bStartStop.Text = "Start";
            this.bStartStop.UseVisualStyleBackColor = true;
            this.bStartStop.Click += new System.EventHandler(this.bStartStop_Click);
            // 
            // bResetLab
            // 
            this.bResetLab.Location = new System.Drawing.Point(332, 98);
            this.bResetLab.Name = "bResetLab";
            this.bResetLab.Size = new System.Drawing.Size(122, 51);
            this.bResetLab.TabIndex = 1;
            this.bResetLab.Text = "Lap/Reset";
            this.bResetLab.UseVisualStyleBackColor = true;
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(66, 42);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(55, 16);
            this.label1.TabIndex = 2;
            this.label1.Text = "00:00:00";
            this.label1.Click += new System.EventHandler(this.label1_Click);
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(8F, 16F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(999, 518);
            this.Controls.Add(this.label1);
            this.Controls.Add(this.bResetLab);
            this.Controls.Add(this.bStartStop);
            this.Name = "Form1";
            this.Text = "Form1";
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Button bStartStop;
        private System.Windows.Forms.Button bResetLab;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.Timer timer1;
    }
}

