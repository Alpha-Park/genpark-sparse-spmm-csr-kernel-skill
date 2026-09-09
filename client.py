class SparseSpMMCSR:
    """
    Compressed Sparse Row (CSR) Sparse Matrix-Dense Matrix Multiplication (SpMM).
    Computes Y = A * X where A is sparse (in CSR format) and X is dense.
    """
    def __init__(self, values, col_indices, row_ptr, num_rows, num_cols):
        self.values = values
        self.col_indices = col_indices
        self.row_ptr = row_ptr
        self.M = num_rows
        self.K = num_cols

    @classmethod
    def from_dense(cls, matrix):
        M = len(matrix)
        K = len(matrix[0]) if M > 0 else 0
        values = []
        col_indices = []
        row_ptr = [0]
        for r in range(M):
            for c in range(K):
                if matrix[r][c] != 0:
                    values.append(matrix[r][c])
                    col_indices.append(c)
            row_ptr.append(len(values))
        return cls(values, col_indices, row_ptr, M, K)

    def multiply_dense(self, X):
        K_x = len(X)
        N = len(X[0])
        assert self.K == K_x, "Inner dimension mismatch"
        Y = [[0.0] * N for _ in range(self.M)]

        for r in range(self.M):
            start = self.row_ptr[r]
            end = self.row_ptr[r + 1]
            for idx in range(start, end):
                val = self.values[idx]
                col = self.col_indices[idx]
                for n in range(N):
                    Y[r][n] += val * X[col][n]

        return Y
