#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;

float expo(float x, float y)
{
    return exp(x);
}

float div(float x, float y)
{
    return x / y;
}

void element_wise_inplace(float *A, float x, size_t m, size_t n, float (*fp)(float, float))
{
    for (size_t i = 0; i < m; i++)
    {
        for (size_t j = 0; j < n; j++)
        {
            A[i*n + j] = fp(A[i*n + j], x);
        }
    }
}

void column_wise_inplace(float *A, const float *x, size_t m, size_t n, float (*fp)(float, float))
{
    for (size_t j = 0; j < n; j++)
    {
        for (size_t i = 0; i < m; i++)
        {
            A[i*n + j] = fp(A[i*n + j], x[i]);
        }
    }
}

void sum_on_axis(const float *X, size_t m, size_t n, size_t axis, float *Dest)
{
    if (axis == 0)
    {
        for (size_t j = 0; j < n; j++)
        {
            Dest[j] = 0;
            for (size_t i = 0; i < m; i++)
            {
                Dest[j] += X[i*n + j];
            }
        }
    }
    else if (axis == 1)
    {
        for (size_t i = 0; i < m; i++)
        {
            Dest[i] = 0;
            for (size_t j = 0; j < n; j++)
            {
                Dest[i] += X[i*n + j];
            }
        }
    }
}

void slice(const float *X, size_t n, size_t start, size_t end, float *Dest)
{
    for (size_t i = 0; i < end - start; i++)
    {
        for (size_t j = 0; j < n; j++)
        {
            Dest[i*n + j] = X[(start+i)*n + j];
        }
    }
}

float inner(const float *u, const float *v, size_t n)
{
    float ret = 0;
    for (size_t i = 0; i < n; i++)
    {
        ret += u[i] * v[i];
    }
    return ret;
}

void matvecmul(const float *X, const float *y, size_t m, size_t n, float *dest)
{
    for (size_t i = 0; i < m; i++)
    {
        dest[i] = inner(X + i * n, y, n);
    }
}

void matmul(const float *A, const float *B, size_t m, size_t n, size_t k, float *Dest)
{
    for (size_t i = 0; i < m; i++)
    {
        for (size_t j = 0; j < k; j++)
        {
            Dest[i*k + j] = 0;
            for (size_t t = 0; t < n; t++)
            {
                Dest[i*k + j] += A[i*n + t] * B[t*k + j];
            }
        }
    }
}

void transpose(const float *A, size_t m, size_t n, float *Dest)
{
    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = 0; j < m; j++)
        {
            Dest[i*m + j] = A[j*n + i];
        }
    }
}

void matsub_inplace(float *A, const float *B, size_t m, size_t n)
{
    for (size_t i = 0; i < m; i++)
    {
        for (size_t j = 0; j < n; j++)
        {
            A[i*n + j] -= B[i*n + j];
        }
    }
}

void matsclmul_inplace(float *A, float x, size_t m, size_t n)
{
    for (size_t i = 0; i < m; i++)
    {
        for (size_t j = 0; j < n; j++)
        {
            A[i*n + j] *= x;
        }
    }
}

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
                                  float *theta, size_t m, size_t n, size_t k,
                                  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    float *H = new float[batch * k];
    float *E = new float[batch * k]();
    float *X_ = new float[batch * n];
    float *X_T = new float[batch * n];
    float *sums = new float[batch];
    float *Grad = new float[n * k];
    for (size_t step = 0; step < m / batch; step++)
    {
        slice(X, n, step * batch, (step+1) * batch, X_);
        matmul(X_, theta, batch, n, k, H);
        element_wise_inplace(H, 0, batch, k, &expo);
        sum_on_axis(H, batch, k, 1, sums);
        column_wise_inplace(H, sums, batch, k, &div);
        // make E rows of one-hot vectors
        for (size_t i = 0; i < batch; i++)
        {
            E[k * i + y[step * batch + i]] = 1;
        }
        matsub_inplace(H, E, batch, k);
        transpose(X_, batch, n, X_T);
        matmul(X_T, H, n, batch, k, Grad);
        matsclmul_inplace(Grad, lr / batch, n, k);
        matsub_inplace(theta, Grad, n, k);
        // reset E to 0
        for (size_t i = 0; i < batch; i++)
        {
            E[k * i + y[step * batch + i]] = 0;
        }
    }
    delete [] H;
    delete [] E;
    delete [] X_;
    delete [] X_T;
    delete [] Grad;
    /// END YOUR CODE
}

/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m)
{
    m.def("softmax_regression_epoch_cpp", [](py::array_t<float, py::array::c_style> X, py::array_t<unsigned char, py::array::c_style> y, py::array_t<float, py::array::c_style> theta, float lr, int batch)
          { softmax_regression_epoch_cpp(
                static_cast<const float *>(X.request().ptr),
                static_cast<const unsigned char *>(y.request().ptr),
                static_cast<float *>(theta.request().ptr),
                X.request().shape[0],
                X.request().shape[1],
                theta.request().shape[1],
                lr,
                batch); }, py::arg("X"), py::arg("y"), py::arg("theta"), py::arg("lr"), py::arg("batch"));
}
