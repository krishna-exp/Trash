import numpy
import sympy


def matmul(a, b, N):
    # computes out = a*b + c
    out = [ 0 for idx in range(N) ]

    for y in range(N):
        sum = out[y]

        for x in range(N):
            sum = sum + (a[y][x] * b[x])

        out[y] = sum

    return out


def add(a, b, N):
    out = [ 0 for _ in range(N) ]

    for idx in range(N):
        out[idx] = a[idx] + b[idx]

    return out


def transpose(a, N):
    out = [ [ 0 for _ in range(N) ] for _ in range(N) ]

    for y in range(N):
        for x in range(N):
            out[y][x] = a[x][y]

    return out


def main():
    # N nodes
    N = 8

    # depth of the pattern
    DEPTH = 3

    # initalizing the output vectors
    # and, the adjacency matrix
    v_sum = [ 0 for _ in range(N) ]
    v = []

    op0 = sympy.symbols('op0')
    op1 = sympy.symbols('op1')
    op2 = sympy.symbols('op2')
    op3 = sympy.symbols('op3')
    op4 = sympy.symbols('op4')
    op5 = sympy.symbols('op5')

    # node vs op mapping
    v.append(op0)
    v.append(op1)
    v.append(op2)
    v.append(op3)
    v.append(op2)
    v.append(op2)
    v.append(op4)
    v.append(op3)

    assert(len(v_sum) == len(v))

    adjacency_matrix = []

    for y in range(N):
        adjacency_vector = []

        for x in range(N):
            adjacency_vector.append(0)

        adjacency_matrix.append(adjacency_vector)

    adjacency_matrix[0][1] = 1
    adjacency_matrix[0][3] = 1
    adjacency_matrix[1][2] = 1
    adjacency_matrix[2][4] = 1
    adjacency_matrix[3][4] = 1
    adjacency_matrix[3][5] = 1
    adjacency_matrix[4][6] = 1
    adjacency_matrix[5][7] = 1

    adjacency_matrix = transpose(adjacency_matrix, N)

    # actual computation

    # matmul is actually a BFS for all the nodes in the graph
    # by adding v_sum back to the 

    print(f'v_sum: {v_sum}')
    print(f'v: {v}')

    print('------------------------------------')

    for itr in range(DEPTH):
        v = matmul(adjacency_matrix, v, N)
        v_sum = add(v_sum, v, N)

        print(f'itr: {itr} => v: {v}')

    print('------------------------------------')

    print(f'v_sum: {v_sum}')
    # print(f'v: {v}')


if __name__ == '__main__':
    main()
