namespace App
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
            this.menuStrip1 = new System.Windows.Forms.MenuStrip();
            this.plikToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.nowyToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.otwórzToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.toolStripSeparator = new System.Windows.Forms.ToolStripSeparator();
            this.zapiszToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.zapiszjakoToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.toolStripSeparator1 = new System.Windows.Forms.ToolStripSeparator();
            this.drukujToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.podglądwydrukuToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.toolStripSeparator2 = new System.Windows.Forms.ToolStripSeparator();
            this.zakończToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.edytujToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.cofnijToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.ponówToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.toolStripSeparator3 = new System.Windows.Forms.ToolStripSeparator();
            this.wytnijToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.kopiujToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.wklejToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.toolStripSeparator4 = new System.Windows.Forms.ToolStripSeparator();
            this.zaznaczwszystkoToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.narzędziaToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.dostosujToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.opcjeToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.pomocToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.zawartośćToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.indeksToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.wyszukajToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.toolStripSeparator5 = new System.Windows.Forms.ToolStripSeparator();
            this.informacjeToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.toolStrip1 = new System.Windows.Forms.ToolStrip();
            this.tsbSave = new System.Windows.Forms.ToolStripButton();
            this.tsbLoad = new System.Windows.Forms.ToolStripButton();
            this.label1 = new System.Windows.Forms.Label();
            this.label2 = new System.Windows.Forms.Label();
            this.label3 = new System.Windows.Forms.Label();
            this.label4 = new System.Windows.Forms.Label();
            this.label5 = new System.Windows.Forms.Label();
            this.label6 = new System.Windows.Forms.Label();
            this.label7 = new System.Windows.Forms.Label();
            this.label8 = new System.Windows.Forms.Label();
            this.label9 = new System.Windows.Forms.Label();
            this.Count = new System.Windows.Forms.NumericUpDown();
            this.PricePerUnit = new System.Windows.Forms.NumericUpDown();
            this.ProdName = new System.Windows.Forms.TextBox();
            this.Producent = new System.Windows.Forms.ComboBox();
            this.Paid = new System.Windows.Forms.CheckBox();
            this.Send = new System.Windows.Forms.CheckBox();
            this.Delivered = new System.Windows.Forms.CheckBox();
            this.DateOfSend = new System.Windows.Forms.DateTimePicker();
            this.BPay = new System.Windows.Forms.Button();
            this.BSend = new System.Windows.Forms.Button();
            this.BDelivered = new System.Windows.Forms.Button();
            this.tbDescription = new System.Windows.Forms.TextBox();
            this.label10 = new System.Windows.Forms.Label();
            this.tbAmount = new System.Windows.Forms.TextBox();
            this.label123 = new System.Windows.Forms.Label();
            this.menuStrip1.SuspendLayout();
            this.toolStrip1.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.Count)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.PricePerUnit)).BeginInit();
            this.SuspendLayout();
            // 
            // menuStrip1
            // 
            this.menuStrip1.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.plikToolStripMenuItem,
            this.edytujToolStripMenuItem,
            this.narzędziaToolStripMenuItem,
            this.pomocToolStripMenuItem});
            this.menuStrip1.Location = new System.Drawing.Point(0, 0);
            this.menuStrip1.Name = "menuStrip1";
            this.menuStrip1.Size = new System.Drawing.Size(800, 27);
            this.menuStrip1.TabIndex = 0;
            this.menuStrip1.Text = "menuStrip1";
            // 
            // plikToolStripMenuItem
            // 
            this.plikToolStripMenuItem.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.nowyToolStripMenuItem,
            this.otwórzToolStripMenuItem,
            this.toolStripSeparator,
            this.zapiszToolStripMenuItem,
            this.zapiszjakoToolStripMenuItem,
            this.toolStripSeparator1,
            this.drukujToolStripMenuItem,
            this.podglądwydrukuToolStripMenuItem,
            this.toolStripSeparator2,
            this.zakończToolStripMenuItem});
            this.plikToolStripMenuItem.Name = "plikToolStripMenuItem";
            this.plikToolStripMenuItem.Size = new System.Drawing.Size(42, 23);
            this.plikToolStripMenuItem.Text = "&Plik";
            this.plikToolStripMenuItem.Click += new System.EventHandler(this.plikToolStripMenuItem_Click);
            // 
            // nowyToolStripMenuItem
            // 
            this.nowyToolStripMenuItem.Image = ((System.Drawing.Image)(resources.GetObject("nowyToolStripMenuItem.Image")));
            this.nowyToolStripMenuItem.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.nowyToolStripMenuItem.Name = "nowyToolStripMenuItem";
            this.nowyToolStripMenuItem.ShortcutKeys = ((System.Windows.Forms.Keys)((System.Windows.Forms.Keys.Control | System.Windows.Forms.Keys.N)));
            this.nowyToolStripMenuItem.Size = new System.Drawing.Size(184, 24);
            this.nowyToolStripMenuItem.Text = "&Nowy";
            this.nowyToolStripMenuItem.Click += new System.EventHandler(this.nowyToolStripMenuItem_Click);
            // 
            // otwórzToolStripMenuItem
            // 
            this.otwórzToolStripMenuItem.Image = ((System.Drawing.Image)(resources.GetObject("otwórzToolStripMenuItem.Image")));
            this.otwórzToolStripMenuItem.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.otwórzToolStripMenuItem.Name = "otwórzToolStripMenuItem";
            this.otwórzToolStripMenuItem.ShortcutKeys = ((System.Windows.Forms.Keys)((System.Windows.Forms.Keys.Control | System.Windows.Forms.Keys.O)));
            this.otwórzToolStripMenuItem.Size = new System.Drawing.Size(184, 24);
            this.otwórzToolStripMenuItem.Text = "&Otwórz";
            this.otwórzToolStripMenuItem.Click += new System.EventHandler(this.otwórzToolStripMenuItem_Click);
            // 
            // toolStripSeparator
            // 
            this.toolStripSeparator.Name = "toolStripSeparator";
            this.toolStripSeparator.Size = new System.Drawing.Size(181, 6);
            // 
            // zapiszToolStripMenuItem
            // 
            this.zapiszToolStripMenuItem.Image = ((System.Drawing.Image)(resources.GetObject("zapiszToolStripMenuItem.Image")));
            this.zapiszToolStripMenuItem.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.zapiszToolStripMenuItem.Name = "zapiszToolStripMenuItem";
            this.zapiszToolStripMenuItem.ShortcutKeys = ((System.Windows.Forms.Keys)((System.Windows.Forms.Keys.Control | System.Windows.Forms.Keys.S)));
            this.zapiszToolStripMenuItem.Size = new System.Drawing.Size(184, 24);
            this.zapiszToolStripMenuItem.Text = "&Zapisz";
            this.zapiszToolStripMenuItem.Click += new System.EventHandler(this.zapiszToolStripMenuItem_Click);
            // 
            // zapiszjakoToolStripMenuItem
            // 
            this.zapiszjakoToolStripMenuItem.Name = "zapiszjakoToolStripMenuItem";
            this.zapiszjakoToolStripMenuItem.Size = new System.Drawing.Size(184, 24);
            this.zapiszjakoToolStripMenuItem.Text = "&Zapisz jako";
            this.zapiszjakoToolStripMenuItem.Click += new System.EventHandler(this.zapiszjakoToolStripMenuItem_Click);
            // 
            // toolStripSeparator1
            // 
            this.toolStripSeparator1.Name = "toolStripSeparator1";
            this.toolStripSeparator1.Size = new System.Drawing.Size(181, 6);
            // 
            // drukujToolStripMenuItem
            // 
            this.drukujToolStripMenuItem.Image = ((System.Drawing.Image)(resources.GetObject("drukujToolStripMenuItem.Image")));
            this.drukujToolStripMenuItem.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.drukujToolStripMenuItem.Name = "drukujToolStripMenuItem";
            this.drukujToolStripMenuItem.ShortcutKeys = ((System.Windows.Forms.Keys)((System.Windows.Forms.Keys.Control | System.Windows.Forms.Keys.P)));
            this.drukujToolStripMenuItem.Size = new System.Drawing.Size(184, 24);
            this.drukujToolStripMenuItem.Text = "&Drukuj";
            this.drukujToolStripMenuItem.Visible = false;
            // 
            // podglądwydrukuToolStripMenuItem
            // 
            this.podglądwydrukuToolStripMenuItem.Image = ((System.Drawing.Image)(resources.GetObject("podglądwydrukuToolStripMenuItem.Image")));
            this.podglądwydrukuToolStripMenuItem.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.podglądwydrukuToolStripMenuItem.Name = "podglądwydrukuToolStripMenuItem";
            this.podglądwydrukuToolStripMenuItem.Size = new System.Drawing.Size(184, 24);
            this.podglądwydrukuToolStripMenuItem.Text = "&Podgląd wydruku";
            this.podglądwydrukuToolStripMenuItem.Visible = false;
            // 
            // toolStripSeparator2
            // 
            this.toolStripSeparator2.Name = "toolStripSeparator2";
            this.toolStripSeparator2.Size = new System.Drawing.Size(181, 6);
            // 
            // zakończToolStripMenuItem
            // 
            this.zakończToolStripMenuItem.Name = "zakończToolStripMenuItem";
            this.zakończToolStripMenuItem.Size = new System.Drawing.Size(184, 24);
            this.zakończToolStripMenuItem.Text = "&Zakończ";
            this.zakończToolStripMenuItem.Click += new System.EventHandler(this.zakończToolStripMenuItem_Click);
            // 
            // edytujToolStripMenuItem
            // 
            this.edytujToolStripMenuItem.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.cofnijToolStripMenuItem,
            this.ponówToolStripMenuItem,
            this.toolStripSeparator3,
            this.wytnijToolStripMenuItem,
            this.kopiujToolStripMenuItem,
            this.wklejToolStripMenuItem,
            this.toolStripSeparator4,
            this.zaznaczwszystkoToolStripMenuItem});
            this.edytujToolStripMenuItem.Name = "edytujToolStripMenuItem";
            this.edytujToolStripMenuItem.Size = new System.Drawing.Size(59, 23);
            this.edytujToolStripMenuItem.Text = "&Edytuj";
            // 
            // cofnijToolStripMenuItem
            // 
            this.cofnijToolStripMenuItem.Name = "cofnijToolStripMenuItem";
            this.cofnijToolStripMenuItem.ShortcutKeys = ((System.Windows.Forms.Keys)((System.Windows.Forms.Keys.Control | System.Windows.Forms.Keys.Z)));
            this.cofnijToolStripMenuItem.Size = new System.Drawing.Size(185, 24);
            this.cofnijToolStripMenuItem.Text = "&Cofnij";
            // 
            // ponówToolStripMenuItem
            // 
            this.ponówToolStripMenuItem.Name = "ponówToolStripMenuItem";
            this.ponówToolStripMenuItem.ShortcutKeys = ((System.Windows.Forms.Keys)((System.Windows.Forms.Keys.Control | System.Windows.Forms.Keys.Y)));
            this.ponówToolStripMenuItem.Size = new System.Drawing.Size(185, 24);
            this.ponówToolStripMenuItem.Text = "&Ponów";
            // 
            // toolStripSeparator3
            // 
            this.toolStripSeparator3.Name = "toolStripSeparator3";
            this.toolStripSeparator3.Size = new System.Drawing.Size(182, 6);
            // 
            // wytnijToolStripMenuItem
            // 
            this.wytnijToolStripMenuItem.Image = ((System.Drawing.Image)(resources.GetObject("wytnijToolStripMenuItem.Image")));
            this.wytnijToolStripMenuItem.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.wytnijToolStripMenuItem.Name = "wytnijToolStripMenuItem";
            this.wytnijToolStripMenuItem.ShortcutKeys = ((System.Windows.Forms.Keys)((System.Windows.Forms.Keys.Control | System.Windows.Forms.Keys.X)));
            this.wytnijToolStripMenuItem.Size = new System.Drawing.Size(185, 24);
            this.wytnijToolStripMenuItem.Text = "Wy&tnij";
            // 
            // kopiujToolStripMenuItem
            // 
            this.kopiujToolStripMenuItem.Image = ((System.Drawing.Image)(resources.GetObject("kopiujToolStripMenuItem.Image")));
            this.kopiujToolStripMenuItem.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.kopiujToolStripMenuItem.Name = "kopiujToolStripMenuItem";
            this.kopiujToolStripMenuItem.ShortcutKeys = ((System.Windows.Forms.Keys)((System.Windows.Forms.Keys.Control | System.Windows.Forms.Keys.C)));
            this.kopiujToolStripMenuItem.Size = new System.Drawing.Size(185, 24);
            this.kopiujToolStripMenuItem.Text = "&Kopiuj";
            // 
            // wklejToolStripMenuItem
            // 
            this.wklejToolStripMenuItem.Image = ((System.Drawing.Image)(resources.GetObject("wklejToolStripMenuItem.Image")));
            this.wklejToolStripMenuItem.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.wklejToolStripMenuItem.Name = "wklejToolStripMenuItem";
            this.wklejToolStripMenuItem.ShortcutKeys = ((System.Windows.Forms.Keys)((System.Windows.Forms.Keys.Control | System.Windows.Forms.Keys.V)));
            this.wklejToolStripMenuItem.Size = new System.Drawing.Size(185, 24);
            this.wklejToolStripMenuItem.Text = "&Wklej";
            // 
            // toolStripSeparator4
            // 
            this.toolStripSeparator4.Name = "toolStripSeparator4";
            this.toolStripSeparator4.Size = new System.Drawing.Size(182, 6);
            // 
            // zaznaczwszystkoToolStripMenuItem
            // 
            this.zaznaczwszystkoToolStripMenuItem.Name = "zaznaczwszystkoToolStripMenuItem";
            this.zaznaczwszystkoToolStripMenuItem.Size = new System.Drawing.Size(185, 24);
            this.zaznaczwszystkoToolStripMenuItem.Text = "&Zaznacz wszystko";
            this.zaznaczwszystkoToolStripMenuItem.Visible = false;
            // 
            // narzędziaToolStripMenuItem
            // 
            this.narzędziaToolStripMenuItem.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.dostosujToolStripMenuItem,
            this.opcjeToolStripMenuItem});
            this.narzędziaToolStripMenuItem.Name = "narzędziaToolStripMenuItem";
            this.narzędziaToolStripMenuItem.Size = new System.Drawing.Size(80, 23);
            this.narzędziaToolStripMenuItem.Text = "&Narzędzia";
            // 
            // dostosujToolStripMenuItem
            // 
            this.dostosujToolStripMenuItem.Enabled = false;
            this.dostosujToolStripMenuItem.Name = "dostosujToolStripMenuItem";
            this.dostosujToolStripMenuItem.Size = new System.Drawing.Size(180, 24);
            this.dostosujToolStripMenuItem.Text = "&Dostosuj";
            // 
            // opcjeToolStripMenuItem
            // 
            this.opcjeToolStripMenuItem.Enabled = false;
            this.opcjeToolStripMenuItem.Name = "opcjeToolStripMenuItem";
            this.opcjeToolStripMenuItem.Size = new System.Drawing.Size(180, 24);
            this.opcjeToolStripMenuItem.Text = "&Opcje";
            // 
            // pomocToolStripMenuItem
            // 
            this.pomocToolStripMenuItem.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.zawartośćToolStripMenuItem,
            this.indeksToolStripMenuItem,
            this.wyszukajToolStripMenuItem,
            this.toolStripSeparator5,
            this.informacjeToolStripMenuItem});
            this.pomocToolStripMenuItem.Name = "pomocToolStripMenuItem";
            this.pomocToolStripMenuItem.Size = new System.Drawing.Size(62, 23);
            this.pomocToolStripMenuItem.Text = "&Pomoc";
            // 
            // zawartośćToolStripMenuItem
            // 
            this.zawartośćToolStripMenuItem.Enabled = false;
            this.zawartośćToolStripMenuItem.Name = "zawartośćToolStripMenuItem";
            this.zawartośćToolStripMenuItem.Size = new System.Drawing.Size(180, 24);
            this.zawartośćToolStripMenuItem.Text = "&Zawartość";
            // 
            // indeksToolStripMenuItem
            // 
            this.indeksToolStripMenuItem.Enabled = false;
            this.indeksToolStripMenuItem.Name = "indeksToolStripMenuItem";
            this.indeksToolStripMenuItem.Size = new System.Drawing.Size(180, 24);
            this.indeksToolStripMenuItem.Text = "&Indeks";
            // 
            // wyszukajToolStripMenuItem
            // 
            this.wyszukajToolStripMenuItem.Enabled = false;
            this.wyszukajToolStripMenuItem.Name = "wyszukajToolStripMenuItem";
            this.wyszukajToolStripMenuItem.Size = new System.Drawing.Size(180, 24);
            this.wyszukajToolStripMenuItem.Text = "&Wyszukaj";
            // 
            // toolStripSeparator5
            // 
            this.toolStripSeparator5.Name = "toolStripSeparator5";
            this.toolStripSeparator5.Size = new System.Drawing.Size(177, 6);
            // 
            // informacjeToolStripMenuItem
            // 
            this.informacjeToolStripMenuItem.Enabled = false;
            this.informacjeToolStripMenuItem.Name = "informacjeToolStripMenuItem";
            this.informacjeToolStripMenuItem.Size = new System.Drawing.Size(180, 24);
            this.informacjeToolStripMenuItem.Text = "&Informacje...";
            // 
            // toolStrip1
            // 
            this.toolStrip1.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.tsbSave,
            this.tsbLoad});
            this.toolStrip1.Location = new System.Drawing.Point(0, 27);
            this.toolStrip1.Name = "toolStrip1";
            this.toolStrip1.Size = new System.Drawing.Size(800, 26);
            this.toolStrip1.TabIndex = 1;
            this.toolStrip1.Text = "toolStrip1";
            // 
            // tsbSave
            // 
            this.tsbSave.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Text;
            this.tsbSave.Image = ((System.Drawing.Image)(resources.GetObject("tsbSave.Image")));
            this.tsbSave.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.tsbSave.Name = "tsbSave";
            this.tsbSave.Size = new System.Drawing.Size(88, 23);
            this.tsbSave.Text = "Save To CSV";
            this.tsbSave.Click += new System.EventHandler(this.tsbSave_Click);
            // 
            // tsbLoad
            // 
            this.tsbLoad.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Text;
            this.tsbLoad.Image = ((System.Drawing.Image)(resources.GetObject("tsbLoad.Image")));
            this.tsbLoad.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.tsbLoad.Name = "tsbLoad";
            this.tsbLoad.Size = new System.Drawing.Size(72, 23);
            this.tsbLoad.Text = "Load CSV";
            this.tsbLoad.Click += new System.EventHandler(this.tsbLoad_Click);
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(35, 66);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(43, 13);
            this.label1.TabIndex = 2;
            this.label1.Text = "Nazwa:";
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(39, 98);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(51, 13);
            this.label2.TabIndex = 3;
            this.label2.Text = "Ilość szt.:";
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Location = new System.Drawing.Point(43, 124);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(35, 13);
            this.label3.TabIndex = 4;
            this.label3.Text = "Cena:";
            this.label3.Click += new System.EventHandler(this.label3_Click);
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Location = new System.Drawing.Point(35, 162);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(50, 13);
            this.label4.TabIndex = 5;
            this.label4.Text = "Wartość:";
            // 
            // label5
            // 
            this.label5.AutoSize = true;
            this.label5.Location = new System.Drawing.Point(35, 202);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(59, 13);
            this.label5.TabIndex = 6;
            this.label5.Text = "Producent:";
            this.label5.Click += new System.EventHandler(this.label5_Click);
            // 
            // label6
            // 
            this.label6.AutoSize = true;
            this.label6.Location = new System.Drawing.Point(35, 257);
            this.label6.Name = "label6";
            this.label6.Size = new System.Drawing.Size(55, 13);
            this.label6.TabIndex = 7;
            this.label6.Text = "Opłacone";
            // 
            // label7
            // 
            this.label7.AutoSize = true;
            this.label7.Location = new System.Drawing.Point(35, 295);
            this.label7.Name = "label7";
            this.label7.Size = new System.Drawing.Size(50, 13);
            this.label7.TabIndex = 8;
            this.label7.Text = "Wysłane";
            // 
            // label8
            // 
            this.label8.AutoSize = true;
            this.label8.Location = new System.Drawing.Point(35, 333);
            this.label8.Name = "label8";
            this.label8.Size = new System.Drawing.Size(71, 13);
            this.label8.TabIndex = 9;
            this.label8.Text = "Data wysyłki:";
            // 
            // label9
            // 
            this.label9.AutoSize = true;
            this.label9.Location = new System.Drawing.Point(35, 385);
            this.label9.Name = "label9";
            this.label9.Size = new System.Drawing.Size(70, 13);
            this.label9.TabIndex = 10;
            this.label9.Text = "Dostarczone:";
            // 
            // Count
            // 
            this.Count.Location = new System.Drawing.Point(102, 91);
            this.Count.Maximum = new decimal(new int[] {
            500,
            0,
            0,
            0});
            this.Count.Name = "Count";
            this.Count.Size = new System.Drawing.Size(120, 20);
            this.Count.TabIndex = 11;
            // 
            // PricePerUnit
            // 
            this.PricePerUnit.Location = new System.Drawing.Point(102, 117);
            this.PricePerUnit.Maximum = new decimal(new int[] {
            3000,
            0,
            0,
            0});
            this.PricePerUnit.Name = "PricePerUnit";
            this.PricePerUnit.RightToLeft = System.Windows.Forms.RightToLeft.Yes;
            this.PricePerUnit.Size = new System.Drawing.Size(100, 20);
            this.PricePerUnit.TabIndex = 12;
            // 
            // ProdName
            // 
            this.ProdName.Location = new System.Drawing.Point(102, 63);
            this.ProdName.Name = "ProdName";
            this.ProdName.Size = new System.Drawing.Size(100, 20);
            this.ProdName.TabIndex = 13;
            // 
            // Producent
            // 
            this.Producent.FormattingEnabled = true;
            this.Producent.Items.AddRange(new object[] {
            "sony",
            "samsung",
            "lg",
            "nokia"});
            this.Producent.Location = new System.Drawing.Point(102, 199);
            this.Producent.Name = "Producent";
            this.Producent.Size = new System.Drawing.Size(121, 21);
            this.Producent.TabIndex = 14;
            // 
            // Paid
            // 
            this.Paid.AutoSize = true;
            this.Paid.Location = new System.Drawing.Point(102, 257);
            this.Paid.Name = "Paid";
            this.Paid.Size = new System.Drawing.Size(15, 14);
            this.Paid.TabIndex = 15;
            this.Paid.UseVisualStyleBackColor = true;
            this.Paid.CheckedChanged += new System.EventHandler(this.checkBox1_CheckedChanged);
            // 
            // Send
            // 
            this.Send.AutoSize = true;
            this.Send.Location = new System.Drawing.Point(102, 295);
            this.Send.Name = "Send";
            this.Send.Size = new System.Drawing.Size(15, 14);
            this.Send.TabIndex = 16;
            this.Send.UseVisualStyleBackColor = true;
            this.Send.CheckedChanged += new System.EventHandler(this.Send_CheckedChanged);
            // 
            // Delivered
            // 
            this.Delivered.AutoSize = true;
            this.Delivered.Location = new System.Drawing.Point(112, 385);
            this.Delivered.Name = "Delivered";
            this.Delivered.Size = new System.Drawing.Size(15, 14);
            this.Delivered.TabIndex = 17;
            this.Delivered.UseVisualStyleBackColor = true;
            // 
            // DateOfSend
            // 
            this.DateOfSend.Format = System.Windows.Forms.DateTimePickerFormat.Short;
            this.DateOfSend.Location = new System.Drawing.Point(112, 333);
            this.DateOfSend.Name = "DateOfSend";
            this.DateOfSend.Size = new System.Drawing.Size(90, 20);
            this.DateOfSend.TabIndex = 18;
            this.DateOfSend.Visible = false;
            // 
            // BPay
            // 
            this.BPay.Location = new System.Drawing.Point(279, 285);
            this.BPay.Name = "BPay";
            this.BPay.Size = new System.Drawing.Size(86, 23);
            this.BPay.TabIndex = 19;
            this.BPay.Text = "Opłacono!";
            this.BPay.UseVisualStyleBackColor = true;
            this.BPay.Click += new System.EventHandler(this.BPay_Click);
            // 
            // BSend
            // 
            this.BSend.Location = new System.Drawing.Point(279, 328);
            this.BSend.Name = "BSend";
            this.BSend.Size = new System.Drawing.Size(86, 23);
            this.BSend.TabIndex = 20;
            this.BSend.Text = "Wyślij!";
            this.BSend.UseVisualStyleBackColor = true;
            this.BSend.Click += new System.EventHandler(this.BSend_Click);
            // 
            // BDelivered
            // 
            this.BDelivered.Location = new System.Drawing.Point(279, 375);
            this.BDelivered.Name = "BDelivered";
            this.BDelivered.Size = new System.Drawing.Size(86, 23);
            this.BDelivered.TabIndex = 21;
            this.BDelivered.Text = "Dostarczono!";
            this.BDelivered.UseVisualStyleBackColor = true;
            this.BDelivered.Click += new System.EventHandler(this.BDelivered_Click);
            // 
            // tbDescription
            // 
            this.tbDescription.Location = new System.Drawing.Point(324, 79);
            this.tbDescription.Multiline = true;
            this.tbDescription.Name = "tbDescription";
            this.tbDescription.Size = new System.Drawing.Size(186, 109);
            this.tbDescription.TabIndex = 22;
            // 
            // label10
            // 
            this.label10.AutoSize = true;
            this.label10.Location = new System.Drawing.Point(324, 63);
            this.label10.Name = "label10";
            this.label10.Size = new System.Drawing.Size(31, 13);
            this.label10.TabIndex = 23;
            this.label10.Text = "Opis:";
            this.label10.Click += new System.EventHandler(this.label10_Click);
            // 
            // tbAmount
            // 
            this.tbAmount.Location = new System.Drawing.Point(102, 162);
            this.tbAmount.Name = "tbAmount";
            this.tbAmount.ReadOnly = true;
            this.tbAmount.Size = new System.Drawing.Size(100, 20);
            this.tbAmount.TabIndex = 24;
            // 
            // label123
            // 
            this.label123.AutoSize = true;
            this.label123.Location = new System.Drawing.Point(206, 119);
            this.label123.Name = "label123";
            this.label123.Size = new System.Drawing.Size(16, 13);
            this.label123.TabIndex = 25;
            this.label123.Text = "zł";
            this.label123.Click += new System.EventHandler(this.label123_Click);
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(800, 450);
            this.Controls.Add(this.label123);
            this.Controls.Add(this.tbAmount);
            this.Controls.Add(this.label10);
            this.Controls.Add(this.tbDescription);
            this.Controls.Add(this.BDelivered);
            this.Controls.Add(this.BSend);
            this.Controls.Add(this.BPay);
            this.Controls.Add(this.DateOfSend);
            this.Controls.Add(this.Delivered);
            this.Controls.Add(this.Send);
            this.Controls.Add(this.Paid);
            this.Controls.Add(this.Producent);
            this.Controls.Add(this.ProdName);
            this.Controls.Add(this.PricePerUnit);
            this.Controls.Add(this.Count);
            this.Controls.Add(this.label9);
            this.Controls.Add(this.label8);
            this.Controls.Add(this.label7);
            this.Controls.Add(this.label6);
            this.Controls.Add(this.label5);
            this.Controls.Add(this.label4);
            this.Controls.Add(this.label3);
            this.Controls.Add(this.label2);
            this.Controls.Add(this.label1);
            this.Controls.Add(this.toolStrip1);
            this.Controls.Add(this.menuStrip1);
            this.MainMenuStrip = this.menuStrip1;
            this.Name = "Form1";
            this.Text = "Form1";
            this.menuStrip1.ResumeLayout(false);
            this.menuStrip1.PerformLayout();
            this.toolStrip1.ResumeLayout(false);
            this.toolStrip1.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.Count)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.PricePerUnit)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.MenuStrip menuStrip1;
        private System.Windows.Forms.ToolStrip toolStrip1;
        private System.Windows.Forms.ToolStripButton tsbSave;
        private System.Windows.Forms.ToolStripButton tsbLoad;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.Label label4;
        private System.Windows.Forms.Label label5;
        private System.Windows.Forms.Label label6;
        private System.Windows.Forms.Label label7;
        private System.Windows.Forms.Label label8;
        private System.Windows.Forms.Label label9;
        private System.Windows.Forms.NumericUpDown Count;
        private System.Windows.Forms.NumericUpDown PricePerUnit;
        private System.Windows.Forms.TextBox ProdName;
        private System.Windows.Forms.ComboBox Producent;
        private System.Windows.Forms.CheckBox Paid;
        private System.Windows.Forms.CheckBox Send;
        private System.Windows.Forms.CheckBox Delivered;
        private System.Windows.Forms.DateTimePicker DateOfSend;
        private System.Windows.Forms.Button BPay;
        private System.Windows.Forms.Button BSend;
        private System.Windows.Forms.Button BDelivered;
        private System.Windows.Forms.TextBox tbDescription;
        private System.Windows.Forms.Label label10;
        private System.Windows.Forms.TextBox tbAmount;
        private System.Windows.Forms.Label label123;
        private System.Windows.Forms.ToolStripMenuItem plikToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem nowyToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem otwórzToolStripMenuItem;
        private System.Windows.Forms.ToolStripSeparator toolStripSeparator;
        private System.Windows.Forms.ToolStripMenuItem zapiszToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem zapiszjakoToolStripMenuItem;
        private System.Windows.Forms.ToolStripSeparator toolStripSeparator1;
        private System.Windows.Forms.ToolStripMenuItem drukujToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem podglądwydrukuToolStripMenuItem;
        private System.Windows.Forms.ToolStripSeparator toolStripSeparator2;
        private System.Windows.Forms.ToolStripMenuItem zakończToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem edytujToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem cofnijToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem ponówToolStripMenuItem;
        private System.Windows.Forms.ToolStripSeparator toolStripSeparator3;
        private System.Windows.Forms.ToolStripMenuItem wytnijToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem kopiujToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem wklejToolStripMenuItem;
        private System.Windows.Forms.ToolStripSeparator toolStripSeparator4;
        private System.Windows.Forms.ToolStripMenuItem zaznaczwszystkoToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem narzędziaToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem dostosujToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem opcjeToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem pomocToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem zawartośćToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem indeksToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem wyszukajToolStripMenuItem;
        private System.Windows.Forms.ToolStripSeparator toolStripSeparator5;
        private System.Windows.Forms.ToolStripMenuItem informacjeToolStripMenuItem;
    }
}

