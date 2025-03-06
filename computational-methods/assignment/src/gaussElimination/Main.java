package gaussElimination;

import java.util.Arrays;

public class Main {
    public static void main(String[] args) {
        double[][] matrixX = {
                {-1, 2, 1},
                {1, -3, -2},
                {3, -1, -1}
        };
        double[][] matrixY = {
                {-1},
                {-1},
                {4}
        };
        double[][] matrix = MatrixConcatHorizontal.concat(matrixX, matrixY);
        double[] solution = GaussElimination.gaussElimination(matrix);

        System.out.println("Rozwiązanie układu równań:");
        System.out.println(Arrays.toString(solution));
    }
}
