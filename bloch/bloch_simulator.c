/* bloch/bloch_simulator.c
 *
 * C-extension entry point that calls blochsimfz(...) defined in bloch.c
 * (This file expects "bloch.c" to be present and contain the actual simulator).
 */

/* Make sure the API version macro is defined *before* including numpy headers */
#define NPY_1_7_API_VERSION 0x00000007
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>

/* Backwards-compatibility: old macro names -> new names */
#ifndef NPY_IN_ARRAY
#define NPY_IN_ARRAY NPY_ARRAY_IN_ARRAY
#endif
#ifndef NPY_OUT_ARRAY
#define NPY_OUT_ARRAY NPY_ARRAY_OUT_ARRAY
#endif
#ifndef NPY_INOUT_ARRAY
#define NPY_INOUT_ARRAY NPY_ARRAY_INOUT_ARRAY
#endif

/* Include the underlying C implementation (original author provided bloch.c) */
#include "bloch.c"

/* Docstrings */
static char module_docstring[] = "Hargreaves Bloch Equation simulator implemented as a C extension for python.";
static char bloch_docstring[] = "Bloch equation simulator.";

/* Prototype for the C simulator function in bloch.c:
   void blochsimfz(double *b1_real, double *b1_imag, double *grx, double *gry,
                   double *grz, double *tp, int ntime, double *t1, double *t2,
                   double *df, int nf, double *dx, double *dy, double *dz,
                   int n_pos, double *mx, double *my, double *mz, int mode);
   (If signature differs in your bloch.c, adjust the call below.)
*/

static PyObject* bloch(PyObject *self, PyObject *args) {
    /* Argument declarations */
    int nf, mode, n_pos;
    PyObject *py_b1_real, *py_b1_imag, *py_grx, *py_gry, *py_grz, *py_t1, *py_t2, *py_tp, *py_df, *py_dx, *py_dy, *py_dz;
    PyObject *py_mx, *py_my, *py_mz;

    /* Arrays: use PyArrayObject* for correct typing */
    PyArrayObject *b1_real_arr = NULL, *b1_imag_arr = NULL, *grx_arr = NULL, *gry_arr = NULL, *grz_arr = NULL;
    PyArrayObject *tp_arr = NULL, *t1_arr = NULL, *t2_arr = NULL, *df_arr = NULL;
    PyArrayObject *dx_arr = NULL, *dy_arr = NULL, *dz_arr = NULL;
    PyArrayObject *mx_arr = NULL, *my_arr = NULL, *mz_arr = NULL;

    int ntime;
    double *b1_real, *b1_imag, *grx, *gry, *grz, *tp, *t1, *t2, *df, *dx, *dy, *dz, *mx, *my, *mz;

    if (!PyArg_ParseTuple(args, "OOOOOOiOOOiOOOiiOOO",
                          &py_b1_real, &py_b1_imag, &py_grx, &py_gry, &py_grz, &py_tp, &ntime,
                          &py_t1, &py_t2, &py_df, &nf, &py_dx, &py_dy, &py_dz, &n_pos, &mode,
                          &py_mx, &py_my, &py_mz)) {
        return NULL;
    }

    /* Convert Python objects to NumPy arrays (double) and cast results */
    b1_real_arr = (PyArrayObject *) PyArray_FROM_OTF(py_b1_real, NPY_DOUBLE, NPY_IN_ARRAY);
    b1_imag_arr = (PyArrayObject *) PyArray_FROM_OTF(py_b1_imag, NPY_DOUBLE, NPY_IN_ARRAY);
    grx_arr = (PyArrayObject *) PyArray_FROM_OTF(py_grx, NPY_DOUBLE, NPY_IN_ARRAY);
    gry_arr = (PyArrayObject *) PyArray_FROM_OTF(py_gry, NPY_DOUBLE, NPY_IN_ARRAY);
    grz_arr = (PyArrayObject *) PyArray_FROM_OTF(py_grz, NPY_DOUBLE, NPY_IN_ARRAY);
    tp_arr = (PyArrayObject *) PyArray_FROM_OTF(py_tp, NPY_DOUBLE, NPY_IN_ARRAY);
    t1_arr = (PyArrayObject *) PyArray_FROM_OTF(py_t1, NPY_DOUBLE, NPY_IN_ARRAY);
    t2_arr = (PyArrayObject *) PyArray_FROM_OTF(py_t2, NPY_DOUBLE, NPY_IN_ARRAY);
    df_arr = (PyArrayObject *) PyArray_FROM_OTF(py_df, NPY_DOUBLE, NPY_IN_ARRAY);
    dx_arr = (PyArrayObject *) PyArray_FROM_OTF(py_dx, NPY_DOUBLE, NPY_IN_ARRAY);
    dy_arr = (PyArrayObject *) PyArray_FROM_OTF(py_dy, NPY_DOUBLE, NPY_IN_ARRAY);
    dz_arr = (PyArrayObject *) PyArray_FROM_OTF(py_dz, NPY_DOUBLE, NPY_IN_ARRAY);
    mx_arr = (PyArrayObject *) PyArray_FROM_OTF(py_mx, NPY_DOUBLE, NPY_INOUT_ARRAY);
    my_arr = (PyArrayObject *) PyArray_FROM_OTF(py_my, NPY_DOUBLE, NPY_INOUT_ARRAY);
    mz_arr = (PyArrayObject *) PyArray_FROM_OTF(py_mz, NPY_DOUBLE, NPY_INOUT_ARRAY);

    /* Check conversions */
    if (!b1_real_arr || !b1_imag_arr || !grx_arr || !gry_arr || !grz_arr ||
        !tp_arr || !t1_arr || !t2_arr || !df_arr || !dx_arr || !dy_arr || !dz_arr ||
        !mx_arr || !my_arr || !mz_arr) {

        Py_XDECREF(b1_real_arr); Py_XDECREF(b1_imag_arr); Py_XDECREF(grx_arr); Py_XDECREF(gry_arr);
        Py_XDECREF(grz_arr); Py_XDECREF(tp_arr); Py_XDECREF(t1_arr); Py_XDECREF(t2_arr);
        Py_XDECREF(df_arr); Py_XDECREF(dx_arr); Py_XDECREF(dy_arr); Py_XDECREF(dz_arr);
        Py_XDECREF(mx_arr); Py_XDECREF(my_arr); Py_XDECREF(mz_arr);

        PyErr_SetString(PyExc_TypeError, "Failed to convert inputs to numpy arrays of type double.");
        return NULL;
    }

    /* Get C pointers to array data */
    b1_real = (double *) PyArray_DATA(b1_real_arr);
    b1_imag = (double *) PyArray_DATA(b1_imag_arr);
    grx = (double *) PyArray_DATA(grx_arr);
    gry = (double *) PyArray_DATA(gry_arr);
    grz = (double *) PyArray_DATA(grz_arr);
    tp = (double *) PyArray_DATA(tp_arr);
    t1 = (double *) PyArray_DATA(t1_arr);
    t2 = (double *) PyArray_DATA(t2_arr);
    df = (double *) PyArray_DATA(df_arr);
    dx = (double *) PyArray_DATA(dx_arr);
    dy = (double *) PyArray_DATA(dy_arr);
    dz = (double *) PyArray_DATA(dz_arr);
    mx = (double *) PyArray_DATA(mx_arr);
    my = (double *) PyArray_DATA(my_arr);
    mz = (double *) PyArray_DATA(mz_arr);

    /* Call the underlying simulator function */
    blochsimfz(b1_real, b1_imag, grx, gry, grz, tp, ntime, t1, t2, df, nf, dx, dy, dz, n_pos, mx, my, mz, mode);

    /* DECREF arrays */
    Py_DECREF(b1_real_arr);
    Py_DECREF(b1_imag_arr);
    Py_DECREF(grx_arr);
    Py_DECREF(gry_arr);
    Py_DECREF(grz_arr);
    Py_DECREF(tp_arr);
    Py_DECREF(t1_arr);
    Py_DECREF(t2_arr);
    Py_DECREF(df_arr);
    Py_DECREF(dx_arr);
    Py_DECREF(dy_arr);
    Py_DECREF(dz_arr);
    Py_DECREF(mx_arr);
    Py_DECREF(my_arr);
    Py_DECREF(mz_arr);

    Py_RETURN_NONE;
}

/* Module method table */
static PyMethodDef module_methods[] = {
    {"bloch_c", bloch, METH_VARARGS, bloch_docstring},
    {NULL, NULL, 0, NULL}
};

/* Module definition */
static struct PyModuleDef bloch_module = {
    PyModuleDef_HEAD_INIT,
    "bloch_simulator",
    module_docstring,
    -1,
    module_methods
};

/* Module initialization */
PyMODINIT_FUNC PyInit_bloch_simulator(void) {
    PyObject *m = PyModule_Create(&bloch_module);
    if (m == NULL) {
        return NULL;
    }
    /* Initialize NumPy C API (must be called after module create) */
    import_array();
    return m;
}

