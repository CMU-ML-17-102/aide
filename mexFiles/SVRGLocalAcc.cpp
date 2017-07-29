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
	hist = SVRGLocalAcc(w, Xt, y, lambda, grad, stepsize, gold, iVals, m, ...
                      kappa, yk, lossType);
	==================================================================
	INPUT PARAMETERS:
	w (d x 1) - initial point; updated in place
	Xt (d x n) - data matrix; transposed (data points are columns); real
	y (n x 1) - labels; in {-1,1}
	lambda - scalar regularization param
	stepSize - a scalar step-size
  gold (d x 1) - full gradient
	iVals (sum(m) x 1) - sequence of examples to choose, between 0 and (n-1)
	m (iters x 1) - sizes of the inner loops
  kappa - regularization parameter of the catalyst
  yk - regularization center of the catalyst
  lossType - integer variable specifying loss type to be worked with 
             1: Quadratic loss
             2: Logistic loss
	==================================================================
	OUTPUT PARAMETERS:
	hist = array of function values after each outer loop.
		   Computed ONLY if explicitely asked for output in MATALB.
*/

/// SVRG_dense runs the SVRGLocalAcc on dense data provided
/// nlhs - number of output parameters requested
///		   if set to 1, function values are computed
/// *prhs[] - array of pointers to the input arguments
mxArray* SVRG_dense(int nlhs, const mxArray *prhs[]) {

	//////////////////////////////////////////////////////////////////
	/// Declare variables ////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////

	// Input variables
	double *w, *Xt, *y;
	double lambda, stepSize;
	long long *iVals, *m;
  double kappa;
  double *yk;
  long lossType;
  
	// Other variables
	long long i; // Some loop indexes
  long k; // Some loop indexes
	long iters; // Number of outer loops
	long long idx; // For choosing indexes
	// Scalar value of the derivative of sigmoid function
	double gradient, gradientold;
	bool evalf = false; // set to true if function values should be evaluated

	double *wold; // Point in which we compute full gradient
	double *gold; // The full gradient in point wold
	double *hist; // Used to store function value at points in history

	mxArray *plhs; // History array to return if needed

	//////////////////////////////////////////////////////////////////
	/// Process input ////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////

	w = mxGetPr(prhs[0]); // The va riable to be learned
	Xt = mxGetPr(prhs[1]); // Data matrix (transposed)
	y = mxGetPr(prhs[2]); // Labels
	lambda = mxGetScalar(prhs[3]); // Regularization parameter
	gold = mxGetPr(prhs[4]); // The outer gradient
	long n, d; // Dimensions of problem
  stepSize = mxGetScalar(prhs[5]); // Step-size (constant)
	iVals = (long long*)mxGetPr(prhs[6]); // Sampled indexes (sampled in advance)
	m = (long long*)mxGetPr(prhs[7]); // Sizes of the inner loops
  kappa = mxGetScalar(prhs[8]); // Acceleration regularization strength
  yk = mxGetPr(prhs[9]); // Point of accelerated regularization
  lossType = (long)mxGetScalar(prhs[10]); // Loss type
	if (nlhs == 1) {
		evalf = true;
	}

	if (!mxIsClass(prhs[6], "int64"))
		mexErrMsgTxt("iVals must be int64");
	if (!mxIsClass(prhs[7], "int64"))
		mexErrMsgTxt("m must be int64");

	//////////////////////////////////////////////////////////////////
	/// Get problem related constants ////////////////////////////////
	//////////////////////////////////////////////////////////////////

	d = mxGetM(prhs[1]); // Number of features, or dimension of problem
	n = mxGetN(prhs[1]); // Number of samples, or data points
	iters = mxGetM(prhs[7]); // Number of outer iterations

	//////////////////////////////////////////////////////////////////
	/// Initialize some values ///////////////////////////////////////
	//////////////////////////////////////////////////////////////////

	// Allocate memory to store full gradient and point in which it
	// was computed
	wold = new double[d];
	if (evalf == true) {
		plhs = mxCreateDoubleMatrix(iters + 1, 1, mxREAL);
		hist = mxGetPr(plhs);
	}

	//////////////////////////////////////////////////////////////////
	/// The SVRG algorithm ///////////////////////////////////////////
	//////////////////////////////////////////////////////////////////

	// The outer loop
	for (k = 0; k < iters; k++)
	{
		// Evaluate function value if output requested
		if (evalf == true) {
			hist[k] = compute_function_value(lossType, w, Xt, y, n, d, lambda);
		}
    
		// Save the point where full gradient was computed; initialize last_seen
    for (i = 0; i < d; i++) { wold[i] = w[i]; }

		// The inner loop
		for (i = 0; i < m[k]; i++) {
      idx = *(iVals++); // Sample function and move pointer

			// Compute current and old scalar gradient of the same example
			gradient = compute_scalar_gradient(lossType, Xt + d*idx, w, y[idx], d);
			gradientold = compute_scalar_gradient(lossType, Xt + d*idx, wold, y[idx], d);

			// Update the test point
			update_test_point_dense_S2GD_acc(Xt + d*idx, w, wold, gold, 
				gradient, gradientold, d, stepSize, lambda, kappa, yk);
		}
	}

	if (evalf == true) {
		hist[iters] = compute_function_value(lossType, w, Xt, y, n, d, lambda);
	}


	//////////////////////////////////////////////////////////////////
	/// Free some memory /////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////

	delete[] wold;
	
	if (evalf == true) { return plhs; }
	else { return 0; }
	
}

/// SVRG_sparse runs the SVRGLocal on sparse data provided
/// nlhs - number of output parameters requested
///		   if set to 1, function values are computed
/// *prhs[] - array of pointers to the input arguments
mxArray* SVRG_sparse(int nlhs, const mxArray *prhs[]) {

	//////////////////////////////////////////////////////////////////
	/// Declare variables ////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////

	// Input variables
	double *w, *Xt, *y;
	double lambda, stepSize;
	long long *iVals, *m;
  double kappa;
  double *yk;
  long lossType;

	// Other variables
	long i, j, k; // Some loop indexes
	long n, d; // Dimensions of problem
	long iters; // Number of outer loops
	long long idx; // For choosing indexes
	// Scalar value of the derivative of objective function
	double gradient, gradientold;
	bool evalf = false; // set to true if function values should be evaluated

	double *wold; // Point in which we compute full gradient
	double *gold; // The full gradient in point wold
	long *last_seen; // used to do lazy "when needed" updates
	double *hist; // Used to store function value at points in history

	mwIndex *ir, *jc; // Used to access nonzero elements of Xt
	mxArray *plhs; // History array to return if needed

	//////////////////////////////////////////////////////////////////
	/// Process input ////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////

	w = mxGetPr(prhs[0]); // The variable to be learned
	Xt = mxGetPr(prhs[1]); // Data matrix (transposed)
	y = mxGetPr(prhs[2]); // Labels
	lambda = mxGetScalar(prhs[3]); // Regularization parameter
	gold = mxGetPr(prhs[4]); // The outer gradient
	stepSize = mxGetScalar(prhs[5]); // Step-size (constant)
	iVals = (long long*)mxGetPr(prhs[6]); // Sampled indexes (sampled in advance)
	m = (long long*)mxGetPr(prhs[7]); // Sizes of the inner loops
  kappa = mxGetScalar(prhs[8]); // Acceleration regularization strength
  yk = mxGetPr(prhs[9]); // Point of accelerated regularization
  lossType = (long)mxGetScalar(prhs[10]); // Loss type
	if (nlhs == 1) {
		evalf = true;
	}

	if (!mxIsClass(prhs[6], "int64"))
		mexErrMsgTxt("iVals must be int64");
	if (!mxIsClass(prhs[7], "int64"))
		mexErrMsgTxt("m must be int64");

	//////////////////////////////////////////////////////////////////
	/// Get problem related constants ////////////////////////////////
	//////////////////////////////////////////////////////////////////

	d = mxGetM(prhs[1]); // Number of features, or dimension of problem
	n = mxGetN(prhs[1]); // Number of samples, or data points
	iters = mxGetM(prhs[7]); // Number of outer iterations
	jc = mxGetJc(prhs[1]); // pointers to starts of columns of Xt
	ir = mxGetIr(prhs[1]); // row indexes of individual elements of Xt

	//////////////////////////////////////////////////////////////////
	/// Initialize some values ///////////////////////////////////////
	//////////////////////////////////////////////////////////////////

	// Allocate memory to store full gradient and point in which it
	// was computed
	wold = new double[d];
	last_seen = new long[d];
	if (evalf == true) {
		plhs = mxCreateDoubleMatrix(iters + 1, 1, mxREAL);
		hist = mxGetPr(plhs);
	}

	//////////////////////////////////////////////////////////////////
	/// The SVRG algorithm ///////////////////////////////////////////
	//////////////////////////////////////////////////////////////////

	// The outer loop
	for (k = 0; k < iters; k++)
	{
		// Evaluate function value if output requested
		if (evalf == true) {
			hist[k] = compute_function_value_sparse(lossType, w, Xt, y, n, d, 
                                              lambda, ir, jc);
		}

		// Save the point where full gradient was computed; initialize last_seen
		for (j = 0; j < d; j++) { wold[j] = w[j]; last_seen[j] = 0;	}

		// The inner loop
		for (i = 0; i < m[k]; i++) {
			idx = *(iVals++); // Sample function and move pointer

			// Update what we didn't in last few iterations
			// Only relevant coordinates
			lazy_update_S2GD_acc(w, wold, gold, last_seen, stepSize, 
                       lambda, i, ir, jc + idx, kappa, yk);

			// Compute current and old scalar sigmoid of the same example
			gradient = compute_scalar_gradient_sparse(lossType, Xt + jc[idx], w, 
              y[idx], jc[idx + 1] - jc[idx], ir + jc[idx]);
			gradientold = compute_scalar_gradient_sparse(lossType, Xt + jc[idx], 
              wold, y[idx], jc[idx + 1] - jc[idx], ir + jc[idx]);

			// Update the test point
			update_test_point_sparse_S2GD(Xt + jc[idx], w, gradient,
				gradientold, jc[idx + 1] - jc[idx], stepSize, ir + jc[idx]);
		}

		// Update the rest of lazy_updates
		finish_lazy_updates_S2GD_acc(w, wold, gold, last_seen, stepSize, 
                                 lambda, m[k], d, kappa, yk);
	}

	// Evaluate the final function value
	if (evalf == true) {
		hist[iters] = compute_function_value_sparse(lossType, w, Xt, y, n, d, 
                                                lambda, ir, jc);
	}

	//////////////////////////////////////////////////////////////////
	/// Free some memory /////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////

	delete[] wold;
	delete[] last_seen;

	//////////////////////////////////////////////////////////////////
	/// Return value /////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////

	if (evalf == true) { return plhs; }
	else { return 0; }

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
	if (mxIsSparse(prhs[1])) {
    plhs[0] = SVRG_sparse(nlhs, prhs);
    // mexErrMsgTxt("I don't speak sparse yet... sorry");
	}
	else {
		plhs[0] = SVRG_dense(nlhs, prhs);
	}
}