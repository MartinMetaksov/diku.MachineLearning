from math import factorial
import numpy as np
from numpy import inner, outer
from numpy.linalg import norm, inv, matrix_rank


def p(v):
    print(v)


a = np.array([1, 2, 2])
b = np.array([3, 2, 1])
M = np.matrix([[1, 0, 0], [0, 4, 0], [0, 0, 2]])

p("\n\nQuestion 1.1. Calculate the inner product (also known as scalar "
  "product or dot product) denoted by ⟨a,b⟩, aTb, or a·b.")
p(inner(a, b))

p("\n\nQuestion 1.2. Calculate the length (also known as Euclidean "
  "norm) |a| of the vector a.")
p(norm(a))
# more on norms
# http://stackoverflow.com/a/9171196/4715690


p("\n\nQuestion 1.3. Calculate the outer product ab^T. Is it equal to "
  "the inner product"
  "a^Tb you computed in Question 1.1?")
p(outer(a, b))

p("\n\nQuestion 1.4. Calculate b^Ta. Is it equal to a^Tb? (Test yourself: "
  "the answer to one of the questions 1.3 and 1.4 should be “yes” and "
  "to the other “no”.)")
p(inner(b, a))
# inner product of a and b = inner product of b and a
# /= outer products in both directions


p("\n\nQuestion 1.5. Calculate the inverse of matrix M, denoted by M^−1. "
  "We remind that you should get that MM^−1 = I, where I is the identity "
  "matrix.")
invM = inv(M)
p(invM)
p("M.M^-1 = I (the identity matrix)")
p(M * invM)

p("\n\nQuestion 1.6. Calculate the matrix-vector product Ma.")
p(M.dot(a))

p("\n\nQuestion 1.7. Let A = ab^T. Calculate the transpose of A, denoted by A^T. "
  "Is A symmetric? (A matrix is called symmetric if A = AT.)")
A = outer(a, b)
p("A")
p(A)
At = A.transpose()
p("At")
p(At)
p("is A symmetric: " + str(np.array_equal(A, At)))

p("\n\nQuestion 1.8. What is the rank of A? (The rank is the number of linearly "
  "independent columns.) Give a short explanation.")
p(matrix_rank(A))
p("Linearly independent columns means that the columns are not multiples of each other. "
  "In this case [1,2,2] * 3 = [3,6,6], which is another column in the matrix."
  "The rank is 1, as [2, 4, 4] and [3, 6, 6] are not multiples of each other.")

p("\n\nQuestion 1.9. What should be the relation between the number of columns and the rank "
  "of a square matrix in order for it to be invertible? Is A = abT invertible?")
try:
    p(inv(A))
except np.linalg.linalg.LinAlgError:
    p("Singular matrix, thus not invertible")
p("The rank of a matrix is zero iff the matrix has 0 non-zero entries. "
  "If a matrix has at least 1 non-zero entry, its rank will be min. 1."
  "The rank must be > 1 in order to be able to invert a matrix, otherwise (if rank is <= 1) "
  "det(A) will be 0 and division by 0 is not allowed.")


# (n over r) = n!/r!(n-r)!
def calc_binomial(n, k):
    return factorial(n) / (factorial(k) * factorial(n - k))


result = 0
for i in range(1, 4):
    result += (calc_binomial(10, i) * pow((1 / 2), (i * (10-i))))

p(result)
