#include "utils.h"
#include <stdlib.h>
#include <math.h>


// these are basically 'imports' from 'basic-sparsem.c' ==> basic implementation
void basic_sparsemm(const COO, const COO, COO *);
void basic_sparsemm_sum(const COO, const COO, const COO,
                        const COO, const COO, const COO,
                        COO *);
void random_matrix(int, int, double, COO *);

// declarations
COO transpose(COO A);
COO multiply(COO A, COO B);
COO add(COO A, COO B);
COO sort(COO);

void my_optimised_sparsemm(const COO A, const COO B, COO *C);
void my_optimised_sparsemm_sum(const COO A, const COO B, const COO C,
                            const COO D, const COO E, const COO F,
                            COO *O);



/* Computes C = A*B.
 * C should be allocated by this routine.
 */
void optimised_sparsemm(const COO A, const COO B, COO *C)
{
	
	// sort the two input matrices
	COO sorted1 = sort(A);
	COO sorted2 = sort(B);

	return my_optimised_sparsemm(sorted1, sorted2, C);
}

void my_optimised_sparsemm(const COO A, const COO B, COO *C)
{
    *C = NULL; 

    *C = multiply(A, B);
}



typedef struct 
{
   int i;
   int j;
   double val;
} triplet;

int comparator(const void *a, const void *b) 
{
	triplet *tripletA = (triplet *)a;
  	triplet *tripletB = (triplet *)b;

	int a_i = tripletA->i;
	int a_j = tripletA->j;
	int b_i = tripletB->i;
	int b_j = tripletB->j;


	if( (a_i < b_i ) || (a_i == b_i && a_j <= b_j))
	{
		return -1;
	}
	else
	{
		return 1;
	}
}


COO sort(COO A)
{

	COO new;

	alloc_sparse(A->m, A->n, A->NZ, &new);

	// array of struct
	triplet array[A->NZ];

	int i;

	//COO to array of structs
	for(i = 0; i < A->NZ; i++)
	{
		triplet trip;

		trip.i = A->coords[i].i;
		trip.j = A->coords[i].j;
		trip.val = A->data[i];

		array[i] = trip;
	}

	// sort the array
	qsort(array, A->NZ, sizeof(triplet), comparator);


	// array back to COO
	int j;
	for(j = 0; j < A->NZ; j++)
	{
		new->coords[j].i = array[j].i;
		new->coords[j].j = array[j].j;
		new->data[j] = array[j].val;
	}
	return new;
}



COO transpose(COO A)
{

	int m = A->m;
	int k = A->n;
	int NZ = A->NZ;

	COO transpose;
	alloc_sparse(m, k, NZ, &transpose);


	int* count;
	count = (int*)calloc(k+1, sizeof(int));
  
  	int i;
    for (i = 0; i < NZ; i++)
    {
    	count[A->coords[i].j]++;
    }
  
  	
    int* index;
	index = (int*)calloc(k+1, sizeof(int));

    index[0] = 0; 
  
    for ( i = 1; i <= k; i++)
    {
    	index[i] = index[i - 1] + count[i - 1]; 
    } 
 	
 	int x;
    for ( x = 0; x < NZ; x++) 
    { 
  
        // insert a data at rpos and increment its value 
        int rpos = index[A->coords[x].j]++; 
  
        // transpose row=col 
        transpose->coords[rpos].i = A->coords[x].j;
  
        // transpose col=row 
        transpose->coords[rpos].j = A->coords[x].i;
  
        // same value 
		transpose->data[rpos] = A->data[x];
	}

	// deallocate memory
	free(count);
	free(index);

    return transpose;
}



// B is transposed 
COO multiply(COO A, COO B)
{

	// transpose the 2nd matrix
	B = transpose(B);

	int m = A->m;
	int k = A->n;

	COO C;

	// allocating space for the product
	int bound = (A->NZ + B->NZ)*5;

	alloc_sparse(m, k, bound, &C);

	int len = 0;

	int apos, bpos; 

	// counter to realloc
	int counter = 0;

    // iterate over all elements of A 
    for (apos = 0; apos < A->NZ;)
    {
    	// row val of apos in A
    	int r = A->coords[apos].i;

    	// loop over all elements of B
    	for(bpos = 0; bpos < B->NZ;)
    	{
    		counter++;

    		/*
    		if(counter == bound)
    		{
    			//realloc
    			alloc_sparse(m, k, bound*2, &C);
    		}
    		*/



    		// column val of bpos in B
    		int c = B->coords[bpos].i;

    		int tempa = apos;
    		int tempb = bpos;

    		// counter (number of values in these arrays)
    		int counter = 0;

    		// create two arrays
    		double listA[A->NZ];
    		double listB[B->NZ];

    		while(tempa < A->NZ && A->coords[tempa].i == r && tempb < B->NZ && B->coords[tempb].i == c)
    		{
    			// if column of val A at tempa < column of val B at temp b 
    			// move down one row in A
    			if(A->coords[tempa].j < B->coords[tempb].j)
    			{
    				tempa++;
    			}
    			else if(A->coords[tempa].j > B->coords[tempb].j)
    			{
    				tempb++;
    			}
    			// A and B have the same column value 
    			else
    			{   
    				listA[counter] = A->data[tempa];
    				listB[counter] = B->data[tempb];
    				counter++;	
    				tempa++;
    				tempb++;
    			}
    		}

    		// calculate the sum
    		double sum = 0.0;
    		int i;

    		// parallelised reduction
    		#pragma acc parallel loop reduction(+:sum)
    		for(i = 0; i < counter; i++)
	    	{
				sum += (listA[i] * listB[i]);
    		}
    		

    		// add new value to c
    		if (fabs(sum) > 1e-15) 
    		{
    			C->coords[len].i = r;
    			C->coords[len].j = c;
				C->data[len] = sum;
				len++;
    		}
    		while (bpos < B->NZ && B->coords[bpos].i == c)
            {
                bpos++; 
            }
    	}

    	while (apos < A->NZ && A->coords[apos].i == r)
        {
	        apos++; 
        }

    }
    
   	C->NZ = len;

   	return C;
}



COO add(COO A, COO B)
{
	int m = A->m;
	int k = A->n;

	COO C;

	// resulting matrix could be MAX m*k size
	alloc_sparse(m, k, m*k, &C);

	int apos = 0;
	int bpos = 0;
	int len = 0;

	while(apos < A->NZ && bpos < B->NZ)
	{
		if(A->coords[apos].i > B->coords[bpos].i ||
			(A->coords[apos].i == B->coords[bpos].i &&
			 A->coords[apos].j > B->coords[bpos].j))
		{
			C->coords[len].i = B->coords[bpos].i;
    		C->coords[len].j = B->coords[bpos].j;
			C->data[len] = B->data[bpos];
			len++;

			bpos++;
		}
		else if(A->coords[apos].i < B->coords[bpos].i ||
			(A->coords[apos].i == B->coords[bpos].i &&
			 A->coords[apos].j < B->coords[bpos].j))
		{
			C->coords[len].i = A->coords[apos].i;
    		C->coords[len].j = A->coords[apos].j;
			C->data[len] = A->data[apos];
			len++;
			apos++;
		}
		else
		{
			double addedVal = A->data[apos] + B->data[bpos];
			

			if (fabs(addedVal) > 1e-15) 
			{
				C->coords[len].i = A->coords[apos].i;
	    		C->coords[len].j = A->coords[apos].j;
				C->data[len] = addedVal;
				len++;
			}
			apos++;
			bpos++;
		}
	}

	while(apos < A->NZ)
	{
		C->coords[len].i = A->coords[apos].i;
	    C->coords[len].j = A->coords[apos].j;
		C->data[len] = A->data[apos++];
		len++;
	}
	while(bpos < B->NZ)
	{
		C->coords[len].i = B->coords[bpos].i;
	    C->coords[len].j = B->coords[bpos].j;
		C->data[len] = B->data[bpos++];
		len++;
	}


	C->NZ = len;
	return C;
}



/* Computes O = (A + B + C) * (D + E + F).
 * O should be allocated by this routine.
 */
void optimised_sparsemm_sum(const COO A, const COO B, const COO C,
                            const COO D, const COO E, const COO F,
                            COO *O)
{
	return my_optimised_sparsemm_sum(A, B, C, D, E, F, O);
}


void my_optimised_sparsemm_sum(const COO A, const COO B, const COO C,
                            const COO D, const COO E, const COO F,
                            COO *O)
{

	*O = NULL;
	COO A_B = add(A, B);
	COO D_E = add(D, E);
	COO AB_C = add(A_B, C);
	COO DE_F = add(D_E, F);
	*O = multiply(AB_C, DE_F);

}






