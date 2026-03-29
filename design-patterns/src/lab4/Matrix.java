package lab4;

import java.util.Arrays;

interface Memento {
}

public class Matrix {
    private double[][] matrix;
    private Storage storage;

    private class MatrixMemento implements Memento {
        int row, col;
        double val_from, val_to;
    }

    public Matrix(int rows, int cols, Storage storage) {
        this.storage = storage;
        matrix = new double[rows][cols];
        double counter = 1.0;

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matrix[i][j] = counter++;
            }
        }

    }

    public void set(int row, int col, double value) {
        if (row < 0 || row >= matrix.length || col < 0 || col >= matrix[0].length) {
            System.out.println("Błąd: Nieprawidłowe indeksy macierzy!");
            return;
        }
        MatrixMemento mem = new MatrixMemento();
        mem.row = row;
        mem.col = col;
        mem.val_from = matrix[row][col];
        mem.val_to = value;
        matrix[row][col] = value;
        storage.save(mem);
    }

    public void undo() {
        MatrixMemento mem = (MatrixMemento) storage.readAndBack();
        if (mem != null) {
            matrix[mem.row][mem.col] = mem.val_from;
            System.out.println("Cofnięto operację.");
        } else {
            System.out.println("Brak operacji do cofnięcia.");
        }

    }

    public void redo() {
        MatrixMemento mem = (MatrixMemento) storage.readForward();
        if (mem != null) {
            matrix[mem.row][mem.col] = mem.val_to;
            System.out.println("Ponowiono operację.");
        }else {
            System.out.println("Brak operacji do ponowienia.");
        }
    }

    @Override
    public String toString() {
        StringBuilder response = new StringBuilder();
        response.append("Aktualny stan macierzy:\n");
        for (double[] row : matrix) {
            for (double col : row) {
                response.append(String.format("%6.1f ", col));
            }
            response.append("\n");
        }
        return response.toString();
    }
}
