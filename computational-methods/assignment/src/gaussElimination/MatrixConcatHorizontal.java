package gaussElimination;

public class MatrixConcatHorizontal {
    public static double[][] concat(double[][] a, double[][] b) {
        int colA = a[0].length;
        int colB = b[0].length;
        double[][] c = new double[a.length][colA + colB];
        for (int i = 0; i < c.length; i++) {
            for (int j = 0; j < a[i].length; j++) {
                c[i][j] = a[i][j];
            }
            int count = 0;
            for (int k = a[i].length; k < c[0].length; k++) {
                c[i][k] = b[i][count];
                count++;
            }
        }
        return c;
    }
}
