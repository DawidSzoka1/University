namespace StoperApp
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
            this.labelTime = new System.Windows.Forms.Label();
            this.bStartStop = new System.Windows.Forms.Button();
            this.bResetLap = new System.Windows.Forms.Button();
            this.listBoxLaps = new System.Windows.Forms.ListBox();
            this.SuspendLayout();
            // 
            // labelTime
            // 
            this.labelTime.AutoSize = true;
            this.labelTime.Location = new System.Drawing.Point(0, 9);
            this.labelTime.Name = "labelTime";
            this.labelTime.Size = new System.Drawing.Size(64, 13);
            this.labelTime.TabIndex = 1;
            this.labelTime.Text = "00:00:00:00";
            // 
            // bStartStop
            // 
            this.bStartStop.Location = new System.Drawing.Point(70, 4);
            this.bStartStop.Name = "bStartStop";
            this.bStartStop.Size = new System.Drawing.Size(75, 23);
            this.bStartStop.TabIndex = 2;
            this.bStartStop.Text = "Start";
            this.bStartStop.UseVisualStyleBackColor = true;
            this.bStartStop.Click += new System.EventHandler(this.bStartStop_Click);
            // 
            // bResetLap
            // 
            this.bResetLap.Location = new System.Drawing.Point(70, 33);
            this.bResetLap.Name = "bResetLap";
            this.bResetLap.Size = new System.Drawing.Size(75, 23);
            this.bResetLap.TabIndex = 3;
            this.bResetLap.Text = "Lap";
            this.bResetLap.UseVisualStyleBackColor = true;
            this.bResetLap.Click += new System.EventHandler(this.button2_Click);
            // 
            // listBoxLaps
            // 
            this.listBoxLaps.FormattingEnabled = true;
            this.listBoxLaps.Location = new System.Drawing.Point(3, 62);
            this.listBoxLaps.Name = "listBoxLaps";
            this.listBoxLaps.Size = new System.Drawing.Size(142, 69);
            this.listBoxLaps.TabIndex = 4;
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(149, 132);
            this.Controls.Add(this.listBoxLaps);
            this.Controls.Add(this.bResetLap);
            this.Controls.Add(this.bStartStop);
            this.Controls.Add(this.labelTime);
            this.Name = "Form1";
            this.Text = "Form1";
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Label labelTime;
        private System.Windows.Forms.Button bStartStop;
        private System.Windows.Forms.Button bResetLap;
        private System.Windows.Forms.ListBox listBoxLaps;
    }
}

