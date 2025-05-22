namespace TreeView
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
            this.treeView1 = new System.Windows.Forms.TreeView();
            this.btnAddRoot = new System.Windows.Forms.Button();
            this.btnAddChild = new System.Windows.Forms.Button();
            this.txtNodeText = new System.Windows.Forms.TextBox();
            this.btnEditText = new System.Windows.Forms.Button();
            this.txtEditText = new System.Windows.Forms.TextBox();
            this.btnDeleteNode = new System.Windows.Forms.Button();
            this.btnDeleteChildren = new System.Windows.Forms.Button();
            this.chkExpandCollapse = new System.Windows.Forms.CheckBox();
            this.menuStrip1 = new System.Windows.Forms.MenuStrip();
            this.fileToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.editToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.toolsToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.helpToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.zapiszToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.wczytajToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.menuStrip1.SuspendLayout();
            this.SuspendLayout();
            // 
            // treeView1
            // 
            this.treeView1.Location = new System.Drawing.Point(31, 58);
            this.treeView1.Name = "treeView1";
            this.treeView1.Size = new System.Drawing.Size(121, 295);
            this.treeView1.TabIndex = 0;
            // 
            // btnAddRoot
            // 
            this.btnAddRoot.Location = new System.Drawing.Point(168, 58);
            this.btnAddRoot.Name = "btnAddRoot";
            this.btnAddRoot.Size = new System.Drawing.Size(141, 23);
            this.btnAddRoot.TabIndex = 1;
            this.btnAddRoot.Text = "Dodaj Korzeń";
            this.btnAddRoot.UseVisualStyleBackColor = true;
            this.btnAddRoot.Click += new System.EventHandler(this.btnAddRoot_Click);
            // 
            // btnAddChild
            // 
            this.btnAddChild.Location = new System.Drawing.Point(168, 98);
            this.btnAddChild.Name = "btnAddChild";
            this.btnAddChild.Size = new System.Drawing.Size(141, 28);
            this.btnAddChild.TabIndex = 2;
            this.btnAddChild.Text = "Dodaj dziecko o tekście:";
            this.btnAddChild.UseVisualStyleBackColor = true;
            this.btnAddChild.Click += new System.EventHandler(this.btnAddChild_Click);
            // 
            // txtNodeText
            // 
            this.txtNodeText.Location = new System.Drawing.Point(324, 98);
            this.txtNodeText.Name = "txtNodeText";
            this.txtNodeText.Size = new System.Drawing.Size(115, 20);
            this.txtNodeText.TabIndex = 3;
            // 
            // btnEditText
            // 
            this.btnEditText.Location = new System.Drawing.Point(168, 132);
            this.btnEditText.Name = "btnEditText";
            this.btnEditText.Size = new System.Drawing.Size(141, 28);
            this.btnEditText.TabIndex = 4;
            this.btnEditText.Text = "Edytuj tekst węzła:";
            this.btnEditText.UseVisualStyleBackColor = true;
            this.btnEditText.Click += new System.EventHandler(this.btnEditText_Click);
            // 
            // txtEditText
            // 
            this.txtEditText.Location = new System.Drawing.Point(324, 140);
            this.txtEditText.Name = "txtEditText";
            this.txtEditText.Size = new System.Drawing.Size(115, 20);
            this.txtEditText.TabIndex = 5;
            // 
            // btnDeleteNode
            // 
            this.btnDeleteNode.Location = new System.Drawing.Point(168, 208);
            this.btnDeleteNode.Name = "btnDeleteNode";
            this.btnDeleteNode.Size = new System.Drawing.Size(271, 30);
            this.btnDeleteNode.TabIndex = 6;
            this.btnDeleteNode.Text = "Usuń wskazany węzeł";
            this.btnDeleteNode.UseVisualStyleBackColor = true;
            this.btnDeleteNode.Click += new System.EventHandler(this.btnDeleteNode_Click);
            // 
            // btnDeleteChildren
            // 
            this.btnDeleteChildren.Location = new System.Drawing.Point(168, 244);
            this.btnDeleteChildren.Name = "btnDeleteChildren";
            this.btnDeleteChildren.Size = new System.Drawing.Size(271, 29);
            this.btnDeleteChildren.TabIndex = 7;
            this.btnDeleteChildren.Text = "Usuń wszystkie węzły potomne";
            this.btnDeleteChildren.UseVisualStyleBackColor = true;
            this.btnDeleteChildren.Click += new System.EventHandler(this.button2_Click);
            // 
            // chkExpandCollapse
            // 
            this.chkExpandCollapse.AutoSize = true;
            this.chkExpandCollapse.Location = new System.Drawing.Point(168, 307);
            this.chkExpandCollapse.Name = "chkExpandCollapse";
            this.chkExpandCollapse.Size = new System.Drawing.Size(133, 17);
            this.chkExpandCollapse.TabIndex = 8;
            this.chkExpandCollapse.Text = "Rozwiń/zwiń wszystko";
            this.chkExpandCollapse.UseVisualStyleBackColor = true;
            this.chkExpandCollapse.CheckedChanged += new System.EventHandler(this.chkExpandCollapse_CheckedChanged);
            // 
            // menuStrip1
            // 
            this.menuStrip1.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.fileToolStripMenuItem,
            this.editToolStripMenuItem,
            this.toolsToolStripMenuItem,
            this.helpToolStripMenuItem});
            this.menuStrip1.Location = new System.Drawing.Point(0, 0);
            this.menuStrip1.Name = "menuStrip1";
            this.menuStrip1.Size = new System.Drawing.Size(800, 27);
            this.menuStrip1.TabIndex = 9;
            this.menuStrip1.Text = "menuStrip1";
            // 
            // fileToolStripMenuItem
            // 
            this.fileToolStripMenuItem.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.zapiszToolStripMenuItem,
            this.wczytajToolStripMenuItem});
            this.fileToolStripMenuItem.Name = "fileToolStripMenuItem";
            this.fileToolStripMenuItem.Size = new System.Drawing.Size(41, 23);
            this.fileToolStripMenuItem.Text = "File";
            // 
            // editToolStripMenuItem
            // 
            this.editToolStripMenuItem.Name = "editToolStripMenuItem";
            this.editToolStripMenuItem.Size = new System.Drawing.Size(44, 23);
            this.editToolStripMenuItem.Text = "Edit";
            // 
            // toolsToolStripMenuItem
            // 
            this.toolsToolStripMenuItem.Name = "toolsToolStripMenuItem";
            this.toolsToolStripMenuItem.Size = new System.Drawing.Size(52, 23);
            this.toolsToolStripMenuItem.Text = "Tools";
            // 
            // helpToolStripMenuItem
            // 
            this.helpToolStripMenuItem.Name = "helpToolStripMenuItem";
            this.helpToolStripMenuItem.Size = new System.Drawing.Size(49, 23);
            this.helpToolStripMenuItem.Text = "Help";
            // 
            // zapiszToolStripMenuItem
            // 
            this.zapiszToolStripMenuItem.Name = "zapiszToolStripMenuItem";
            this.zapiszToolStripMenuItem.Size = new System.Drawing.Size(180, 24);
            this.zapiszToolStripMenuItem.Text = "zapisz";
            this.zapiszToolStripMenuItem.Click += new System.EventHandler(this.zapiszToolStripMenuItem_Click);
            // 
            // wczytajToolStripMenuItem
            // 
            this.wczytajToolStripMenuItem.Name = "wczytajToolStripMenuItem";
            this.wczytajToolStripMenuItem.Size = new System.Drawing.Size(180, 24);
            this.wczytajToolStripMenuItem.Text = "wczytaj";
            this.wczytajToolStripMenuItem.Click += new System.EventHandler(this.wczytajToolStripMenuItem_Click);
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(800, 450);
            this.Controls.Add(this.chkExpandCollapse);
            this.Controls.Add(this.btnDeleteChildren);
            this.Controls.Add(this.btnDeleteNode);
            this.Controls.Add(this.txtEditText);
            this.Controls.Add(this.btnEditText);
            this.Controls.Add(this.txtNodeText);
            this.Controls.Add(this.btnAddChild);
            this.Controls.Add(this.btnAddRoot);
            this.Controls.Add(this.treeView1);
            this.Controls.Add(this.menuStrip1);
            this.MainMenuStrip = this.menuStrip1;
            this.Name = "Form1";
            this.Text = "Form1";
            this.menuStrip1.ResumeLayout(false);
            this.menuStrip1.PerformLayout();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.TreeView treeView1;
        private System.Windows.Forms.Button btnAddRoot;
        private System.Windows.Forms.Button btnAddChild;
        private System.Windows.Forms.TextBox txtNodeText;
        private System.Windows.Forms.Button btnEditText;
        private System.Windows.Forms.TextBox txtEditText;
        private System.Windows.Forms.Button btnDeleteNode;
        private System.Windows.Forms.Button btnDeleteChildren;
        private System.Windows.Forms.CheckBox chkExpandCollapse;
        private System.Windows.Forms.MenuStrip menuStrip1;
        private System.Windows.Forms.ToolStripMenuItem fileToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem editToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem toolsToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem helpToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem zapiszToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem wczytajToolStripMenuItem;
    }
}

