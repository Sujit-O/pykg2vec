#include <stdio.h>
#include <stdlib.h>

// Function to read the data
__declspec(dllexport) float *read_data(int batch_idx, int batch_size, int total_entity, char *filename)
{
   //return multidimensional array as python list
   int MAXCHAR;
   int i;
   int j;
   float tmp;
   int k;
   int len;
   FILE *fp;
   char ch[200000];
   float *arr;
   //open the file pointer
    fp = fopen(filename, "r");
    if (fp == NULL)
        {
            printf("Could not open file %s \n",filename);
            return NULL;
        }

    //testing codes
//    printf("filename: %s \n", filename);
//    printf("batch_idx:%d \n", batch_idx);
//    printf("batch_size:%d \n", batch_size);
//    printf("total_entity:%d \n", total_entity);
    //assing the maximum array length
    MAXCHAR = (total_entity*2+4)*batch_size;
    arr =(float *)malloc(sizeof(float *) * MAXCHAR);
    if (arr == NULL) {
         printf("Could not assign memory! \n");
        return NULL;
    }

    //get the total line size to find offset
    fscanf(fp,"%[^\n]", ch);
//    printf("Data from the file:%s \n", ch);
    len = ftell(fp);
//    printf("Total size of reading one line = %d bytes\n", len);

    fseek(fp, 0, SEEK_SET);
    //Change the file point to the desired batch idx
    //add \r and \n (2 bytes to size of each line)
    len--;
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
                    if(fscanf(fp, "%f", &arr[k]) != 1)
                        {
                            printf("no value read! \n");
                            continue;
                        }
                    else
                        {

//                            printf("i:%d, j:%d, value:%f \t",i,j, arr[k]);
                            k++;
                        }
                 }
//                 printf("\n");
             }
        }
    fclose(fp);
    return arr;
}
