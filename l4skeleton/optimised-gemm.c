
// Christian Johnston
// Optimising matrix-matrix multiplications
// BLIS Realisation


// ****** IMPORTANT ****** //
// Code makes use of work from:
// R. A. van de Geijn. “Anatomy of high-performance matrix multiplication”
// F. G. V. Zee and R. A. van de Geijn. “BLIS: a framework for rapidly instantiating BLAS functionality”
// R. A. Robert van de Geijn- The GoToBLAS/BLIS Approach to Optimizing Matrix-Matrix Multiplication – Step-by-Step



// Macro definitions
#define A(i,j) a[ (j)*lda + (i) ]
#define B(i,j) b[ (j)*ldb + (i) ]
#define C(i,j) c[ (j)*ldc + (i) ]
#define X(i) x[ (i)*incx ]
#define min( i, j ) ( (i)<(j) ? (i): (j) )

//inner kernel size
int m_r = 4;
int n_r = 4;

// block sizes
int k_c = 256;
int m_c = 512;

// function declarations
void InnerKernel(int m, int n, int k, const double *a, int lda, const double *b, int ldb, double *c, int ldc);
void PackMatrixA( int k, const double *a, int lda, double *a_to);
void PackMatrixB( int k, const double *b, int ldb, double *b_to);
void AddDot1x4( int k, const double *a, int lda,  const double *b, int ldb, double *c, int ldc);
void AddDotMRxNR( int k, const double *a, int lda, const double *b, int ldb, double *c, int ldc);
void basic_gemm(int, int, int, const double *, int, const double *, int, double *, int);
void AddDot( int, const double *, int, const double *, double * );


// MAIN
void optimised_gemm(int m, int n, int k, const double *a, int lda, const double *b, int ldb, double *c, int ldc)
{
	int i, p, pb, ib;
	for (p=0; p<k; p+=k_c)
	{
  		pb = min(k-p, k_c);
  		for (i=0; i<m; i+=m_c)
  		{
  			ib = min( m-i, m_c );
  			InnerKernel( ib, n, pb, &A( i,p ), lda, &B(p, 0 ), ldb, &C( i,0 ), ldc );
    	}
  	}
}



void InnerKernel(int m, int n, int k, const double *a, int lda, const double *b, int ldb, double *c, int ldc)
{

	double packedA[m * k], packedB[k * n];

	int i, j;
	for( j=0; j<n-m_r; j+=m_r )
	{
		PackMatrixB( k, &B( 0, j ), ldb, &packedB[ j*k ] );
    	for ( i=0; i<m-n_r; i+=n_r )
    	{
    		if ( j == 0 ) PackMatrixA( k, &A( i, 0 ), lda, &packedA[ i*k ] );
      		AddDotMRxNR( k, &packedA[ i*k ], 4, &B( 0,j ), ldb, &C( i,j ), ldc );
    	}

    	for(; i<m; i++ )
    	{      
      		AddDot1x4( k, &A( i,0 ), lda, &B( 0,j ), ldb, &C( i,j ), ldc );
    	}
  	}
  	for(; j<n; j++)
  	{
  		for (i = 0; i < m; i++) 
	    {
	        AddDot( k, &A( i,0 ), lda, &B( 0,j ), &C( i,j ) );
	    }
  	}
}

void PackMatrixA( int k, const double *a, int lda, double *a_to )
{
	int j;

	/* loop over columns of A */
  	for( j=0; j<k; j++)
  	{  
    	const double *a_ij_pntr = &A( 0, j );

	    *a_to++ = *a_ij_pntr;
	    *a_to++ = *(a_ij_pntr+1);
	    *a_to++ = *(a_ij_pntr+2);
	    *a_to++ = *(a_ij_pntr+3);
  	}
}

void PackMatrixB( int k, const double *b, int ldb, double *b_to )
{
	int i;
  	const double 
    	*b_i0_pntr = &B( 0, 0 ), *b_i1_pntr = &B( 0, 1 ),
    	*b_i2_pntr = &B( 0, 2 ), *b_i3_pntr = &B( 0, 3 );

  	for( i=0; i<k; i++)
  	{  
	    *b_to++ = *b_i0_pntr++;
	    *b_to++ = *b_i1_pntr++;
	    *b_to++ = *b_i2_pntr++;
	    *b_to++ = *b_i3_pntr++;
  	}
}

// is this 1xMR or 1xNR
void AddDot1x4( int k, const double *a, int lda,  const double *b, int ldb, double *c, int ldc )
{
	int p;
  	for ( p=0; p<k; p++ )
  	{
	    C( 0, 0 ) += A( 0, p ) * B( p, 0 ); 
	    C( 0, 1 ) += A( 0, p ) * B( p, 1 );
	    C( 0, 2 ) += A( 0, p ) * B( p, 2 ); 
	    C( 0, 3 ) += A( 0, p ) * B( p, 3 );    
  	}
}

void AddDotMRxNR( int k, const double *a, int lda, const double *b, int ldb, double *c, int ldc )
{
	int p;

  	for (p=0; p<k; p++ )
  	{
  		int q;

  		for(q=0; q<m_r; q++)
  		{
  			int r;
  			for(r=0; r<n_r; r++)
  			{
  				C( q, r ) += A( q, p ) * B( p, r );     
  			}
  		}
  	}
}

void AddDot( int k, const double *x, int incx,  const double *y, double *gamma )
{
	int p;

  	for ( p=0; p<k; p++ )
  	{
    	*gamma += X( p ) * y[ p ];     
  	}
}
