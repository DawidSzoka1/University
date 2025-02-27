import javax.swing.*;
import java.awt.*;

public class AddNewCurvePanel extends JPanel {
    JTextField p0 = new JTextField(5);
    JLabel labelp0 = new JLabel("p0 np 10:20");
    JTextField p1 = new JTextField(5);
    JLabel labelp1 = new JLabel("p1 np 10:20");
    JTextField pk0 = new JTextField(5);
    JLabel labelpk0 = new JLabel("pk0 np 10:20");
    JTextField pk1 = new JTextField(5);
    JLabel labelpk1 = new JLabel("pk1 np 10:20");
    JButton create = new JButton("Add curve");
    AddNewCurvePanel(){
        setLayout(new FlowLayout());
        add(labelp0);
        add(p0);
        add(labelp1);
        add(p1);
        add(labelpk0);
        add(pk0);
        add(labelpk1);
        add(pk1);
        add(create);
    }

}
