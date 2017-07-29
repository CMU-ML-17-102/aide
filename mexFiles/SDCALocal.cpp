///  Copyright [2014] [Jakub Konecny]
///
/// Licensed under the Apache License, Version 2.0 (the "License");
/// you may not use this file except in compliance with the License.
/// You may obtain a copy of the License at
/// 
///     http://www.apache.org/licenses/LICENSE-2.0
/// 
/// Unless required by applicable law or agreed to in writing, software
/// distributed under the License is distributed on an "AS IS" BASIS,
/// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
/// See the License for the specific language governing permissions and
/// limitations under the License.

//////////////////////////////////////////////////////////////////////
/// Jakub Konecny | www.jakubkonecny.com /////////////////////////////
/// last update : 8 July 2014            /////////////////////////////
//////////////////////////////////////////////////////////////////////

#include <math.h>
#include "mex.h"
#include <string.h>

#include "utilities_dense.h"
#include "utilities_sparse.h"

/*
	USAGE:
	hist = SDCALocal(w, Xt, y, norms, lambda, iVals, nTotal);
	==================================================================
	INPUT PARAMETERS:
	w (d x 1) - initial point; updated in place
	Xt (d x n) - data matrix; transposed (data points are columns); real
	y (n x 1) - labels; in {-1,1}
	norms (n x 1) - norms of the data examples in Xt
	lambda - scalar regularization param
  iVals (m x 1) - sequence of examples to choose, between 0 and (n-1)
	nTotal - total number of data points in the problem
  ==================================================================
	OUTPUT PARAMETERS:
	hist = array of function values after each outer loop.
		   Computed ONLY if explicitely asked for output in MATALB.
*/

/// SDCALocal_dense runs the local SDCA on dense data provided
/// nlhs - number of output parameters requested
///		   if set to 1, function values are computed
/// *prhs[] - array of pointers to the input arguments
mxArray* SDCALocal_dense(int nlhs, const mxArray *prhs[]) {

	//////////////////////////////////////////////////////////////////
	/// Declare variables ////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////

	// Input variables
	double *w, *alpha, *Xt, *y, *norms;
	double lambda;
	long long *iVals;
  double nTotal;
  long lossType;
  
	// Other variables
	long long i; // Some loop indexes
  long long idx; // For choosing indexes
	long k; // Some loop indexes
	long n, d; // Dimensions of problem
	long iters; // Number of outer loops
  double update; // To store SDCA step
	
	//////////////////////////////////////////////////////////////////
	/// Process input ////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////

	w = mxGetPr(prhs[0]); // The variable to be learned
	alpha = mxGetPr(prhs[1]); // The local dual variables
  Xt = mxGetPr(prhs[2]); // Data matrix (transposed)
	y = mxGetPr(prhs[3]); // Labels
  norms = mxGetPr(prhs[4]); // norms of data points
  lambda = mxGetScalar(prhs[5]); // Regularization parameter
	iVals = (long long*)mxGetPr(prhs[6]); // Sampled indexes (sampled in advance)
	nTotal = mxGetScalar(prhs[7]); // Total number of data poitns
  lossType = (long)mxGetScalar(prhs[8]); // Loss type
  
	if (!mxIsClass(prhs[6], "int64"))
		mexErrMsgTxt("iVals must be int64");
	
	//////////////////////////////////////////////////////////////////
	/// Get problem related constants ////////////////////////////////
	//////////////////////////////////////////////////////////////////

	d = mxGetM(prhs[2]); // Number of features, or dimension of problem
	n = mxGetN(prhs[2]); // Number of samples, or data points
	iters = mxGetM(prhs[6]); // Number of outer iterations
  
  double lambda_n = lambda * nTotal; // constant used in algorithm

	//////////////////////////////////////////////////////////////////
	/// The local SDCA  //////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////

	// Local SDCA iterations
	for (i = 0; i < iters; i++)
	{
    idx = *(iVals++); // Sample function and move pointer
    
    // Compute the SDCA update step
    update = SDCA_step(lossType, w, alpha[idx], Xt + d*idx, y[idx], lambda_n, 
                       norms[idx], d);
    
    // update dual variable
    alpha[idx] += update;
    // update primal variable
    update_primal(w, Xt + d*idx, update, d, lambda_n);
  }
  
	return 0;
	
}

/// SDCALocal_sparse runs the local SDCA on sparse data provided
/// nlhs - number of output parameters requested
///		   if set to 1, function values are computed
/// *prhs[] - array of pointers to the input arguments
mxArray* SDCALocal_sparse(int nlhs, const mxArray *prhs[]) {
  
  //////////////////////////////////////////////////////////////////
	/// Declare variables ////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////

	// Input variables
	double *w, *alpha, *Xt, *y, *norms;
	double lambda, nTotal;
	long long *iVals, *m;
  long lossType;

	// Other variables
	long long i; // Some loop indexes
  long j, k; // Some more loop indexes
	long n, d; // Dimensions of problem
	long iters; // Number of outer loops
	long long idx; // For choosing indexes
  double update; // To store SDCA step
  
  mwIndex *ir, *jc; // Used to access nonzero elements of Xt
	mxArray *plhs; // History array to return if needed

	//////////////////////////////////////////////////////////////////
	/// Process input ////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////

	w = mxGetPr(prhs[0]); // The variable to be learned
	alpha = mxGetPr(prhs[1]); // The local dual variables
  Xt = mxGetPr(prhs[2]); // Data matrix (transposed)
	y = mxGetPr(prhs[3]); // Labels
  norms = mxGetPr(prhs[4]); // norms of data points
  lambda = mxGetScalar(prhs[5]); // Regularization parameter
	iVals = (long long*)mxGetPr(prhs[6]); // Sampled indexes (sampled in advance)
	nTotal = mxGetScalar(prhs[7]); // Total number of data poitns
  lossType = (long)mxGetScalar(prhs[8]); // Loss type
  
	if (!mxIsClass(prhs[6], "int64"))
		mexErrMsgTxt("iVals must be int64");

	//////////////////////////////////////////////////////////////////
	/// Get problem related constants ////////////////////////////////
	//////////////////////////////////////////////////////////////////

	d = mxGetM(prhs[2]); // Number of features, or dimension of problem
	n = mxGetN(prhs[2]); // Number of samples, or data points
	iters = mxGetM(prhs[6]); // Number of outer iterations
	jc = mxGetJc(prhs[2]); // pointers to starts of columns of Xt
	ir = mxGetIr(prhs[2]); // row indexes of individual elements of Xt
  
  double lambda_n = lambda * nTotal; // constant used in algorithm

	//////////////////////////////////////////////////////////////////
	/// The local SDCA ///////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////

	// Local SDCA iterations
	for (k = 0; k < iters; k++)
	{
    idx = *(iVals++); // Sample function and move pointer
    
    // Compute the SDCA update step
    update = SDCA_step_sparse(lossType, w, alpha[idx], Xt + jc[idx], 
        y[idx], lambda_n, norms[idx], jc[idx + 1] - jc[idx], ir + jc[idx]);
    
    // update dual variable
    alpha[idx] += update;
    // update primal variable
    update_primal_sparse(w, Xt + jc[idx], update, jc[idx + 1] - jc[idx], 
                         lambda_n, ir + jc[idx]);
	}
  
  return 0;
  
}

/// Entry function of MATLAB
/// nlhs - number of output parameters
/// *plhs[] - array poiters to the outputs
/// nrhs - number of input parameters
/// *prhs[] - array of pointers to inputs
/// For more info about this syntax see 
/// http://www.mathworks.co.uk/help/matlab/matlab_external/gateway-routine.html
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

	// First determine, whether the data matrix is stored in sparse format.
	// If it is, use more efficient algorithm
	if (mxIsSparse(prhs[2])) {
    plhs[0] = SDCALocal_sparse(nlhs, prhs);
    // mexErrMsgTxt("I don't speak sparse yet... sorry");
	}
	else {
		plhs[0] = SDCALocal_dense(nlhs, prhs);
    // mexErrMsgTxt("UNDER CONSTRUCTION! Do not touch me yet please.");
	}
}