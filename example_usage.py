from client import SparseSpMMCSR

def main():
    print("=== Testing CSR Sparse Matrix Multiplication (SpMM) ===")
    sparse_A = [
        [10.0,  0.0,  0.0],
        [ 0.0, 20.0, 30.0],
        [ 0.0,  0.0, 40.0]
    ]
    sp = SparseSpMMCSR.from_dense(sparse_A)
    print(f"CSR representation: non-zeros={len(sp.values)}, row_ptr={sp.row_ptr}")

    dense_X = [
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0]
    ]
    Y = sp.multiply_dense(dense_X)
    print("SpMM Result Y = A * X:")
    for row in Y:
        print(" ", row)

    assert Y == [[10.0, 20.0], [210.0, 260.0], [200.0, 240.0]]
    print("=== All tests passed successfully! ===")

if __name__ == "__main__":
    main()
