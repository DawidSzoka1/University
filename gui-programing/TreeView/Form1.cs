using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.IO;


namespace TreeView
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }

        private void button2_Click(object sender, EventArgs e)
        {
            if (treeView1.SelectedNode != null)
            {
                treeView1.SelectedNode.Nodes.Clear();
            }
        }

        private void btnAddRoot_Click(object sender, EventArgs e)
        {
            treeView1.Nodes.Add("Nowy korzeń");
        }

        private void btnAddChild_Click(object sender, EventArgs e)
        {
            if (treeView1.SelectedNode != null && !string.IsNullOrWhiteSpace(txtNodeText.Text))
            {
                treeView1.SelectedNode.Nodes.Add(txtNodeText.Text);
                treeView1.SelectedNode.Expand();
            }
        }

        private void btnEditText_Click(object sender, EventArgs e)
        {
            if (treeView1.SelectedNode != null && !string.IsNullOrWhiteSpace(txtEditText.Text))
            {
                treeView1.SelectedNode.Text = txtEditText.Text;
            }
        }

        private void btnDeleteNode_Click(object sender, EventArgs e)
        {
            if (treeView1.SelectedNode != null)
            {
                treeView1.SelectedNode.Remove();
            }
        }

        private void chkExpandCollapse_CheckedChanged(object sender, EventArgs e)
        {
            if (chkExpandCollapse.Checked)
                treeView1.ExpandAll();
            else
                treeView1.CollapseAll();
        }
        private string ToNewick(TreeNode node)
        {
            if (node.Nodes.Count == 0)
                return node.Text;

            List<string> children = new List<string>();
            foreach (TreeNode child in node.Nodes)
                children.Add(ToNewick(child));

            return $"({string.Join(",", children)}){node.Text}";
        }

        private string TreeToNewick(System.Windows.Forms.TreeView treeView)
        {
            List<string> roots = new List<string>();
            foreach (TreeNode node in treeView.Nodes)
                roots.Add(ToNewick(node));
            return string.Join(";", roots) + ";";
        }

        private TreeNode ParseNewick(ref string text)
        {
            TreeNode node = new TreeNode();
            if (text.StartsWith("("))
            {
                text = text.Substring(1);
                while (!text.StartsWith(")"))
                {
                    TreeNode child = ParseNewick(ref text);
                    node.Nodes.Add(child);
                    if (text.StartsWith(",")) text = text.Substring(1);
                }
                text = text.Substring(1); // usuń ')'
            }

            int i = 0;
            while (i < text.Length && text[i] != ',' && text[i] != ')' && text[i] != ';')
                i++;

            node.Text = text.Substring(0, i);
            text = text.Substring(i);
            return node;
        }

        private void FromNewick(string newick)
        {
            treeView1.Nodes.Clear();
            newick = newick.TrimEnd(';');

            while (!string.IsNullOrEmpty(newick))
            {
                TreeNode node = ParseNewick(ref newick);
                treeView1.Nodes.Add(node);
                if (newick.StartsWith(";")) break;
                if (newick.StartsWith(",")) newick = newick.Substring(1);
            }

            treeView1.ExpandAll();
        }

        private void zapiszToolStripMenuItem_Click(object sender, EventArgs e)
        {
            SaveFileDialog sfd = new SaveFileDialog();
            sfd.Filter = "Plik Newick|*.txt";

            if (sfd.ShowDialog() == DialogResult.OK)
            {
                string newick = TreeToNewick(treeView1);
                File.WriteAllText(sfd.FileName, newick);
            }
        }

        private void wczytajToolStripMenuItem_Click(object sender, EventArgs e)
        {
            OpenFileDialog ofd = new OpenFileDialog();
            ofd.Filter = "Plik Newick|*.txt";

            if (ofd.ShowDialog() == DialogResult.OK)
            {
                string text = File.ReadAllText(ofd.FileName);
                FromNewick(text);
            }
        }
    }
}
