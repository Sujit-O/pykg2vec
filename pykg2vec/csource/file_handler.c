#include <stdio.h>
#include <Python.h>

// Function to read the data
static PyObject* read_data(PyObject* self, PyObject* args)
{
   //return multidimensional array as python list
   PyObject *pylist;
   int batch_idx;
   int batch_size;
   int total_entity;
   int MAXCHAR;
   char *filename;
   int i;
   int j;
   float tmp;
   int k;
   int len;
   FILE *fp;
   char ch[200000];

   if (!PyArg_ParseTuple(args, "iiis", &batch_idx, &batch_size, &total_entity, &filename)) {
      return NULL;
   }

    //assing the maximum array length
    MAXCHAR = (total_entity*2+4)*batch_size;
    pylist = PyList_New(MAXCHAR);

    //open the file
    fp = fopen(filename, "r");
    fscanf(fp,"%[^\n]", ch);
    len = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    //Change the file point to the desired batch idx
    //add \r and \n (2 bytes to size of each line)
    fseek(fp, batch_idx*(len+2)*batch_size, SEEK_CUR);
    if (fp == NULL)
        {
            printf("Could not open file %s \n",filename);
        }
    else
        {
        k=0;
        for(i=0; i<batch_size; i++)
            {
             for(j=0; j<(total_entity*2+4); j++)
                 {
                    if(fscanf(fp, "%f", &tmp) != 1)
                        {
                            printf("no value read! \n");
                            continue;
                        }
                    else
                        {
                            PyList_SET_ITEM(pylist, k, PyFloat_FromDouble(tmp));
                            k++;
//                            printf("i:%d, j:%d, value:%f \t",i,j, tmp);
                        }
                 }
//                 printf("\n");
             }
        }
    fclose(fp);
    return pylist;
}

// Our Module's Function Definition struct
// We require this `NULL` to signal the end of our method
// definition
static PyMethodDef myMethods[] = {
    { "read_data", read_data, METH_VARARGS, "Reads Data from the Disk" },
    { NULL, NULL, 0, NULL }
};

// Our Module Definition struct
static struct PyModuleDef file_handler = {
    PyModuleDef_HEAD_INIT,
    "file_handler",
    "File Handler in C",
    -1,
    myMethods
};

// Initializes our module using our above struct
PyMODINIT_FUNC PyInit_file_handler(void)
{
    return PyModule_Create(&file_handler);
}