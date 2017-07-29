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

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//// This header contains helper functions for algorithms         ////
//// included in experiments for the S2GD paper                   ////
//// (Semi-Stochastic Gradient Descent Methods)                   ////
//// This contains functions for sparse data matrix               ////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

/// For more info about ir/jc convention below, see "Sparse Matrices" in 
/// http://www.mathworks.co.uk/help/matlab/matlab_external/matlab-data.html

// Declarations
double compute_function_value_sparse(long lossType, double* w, double *Xt, double *y,
	long n, long d, double lambda, mwIndex *ir, mwIndex *jc);
void compute_full_gradient_sparse(long lossType, double *Xt, double *w, double *y, 
  double *g, long n, long d, double lambda, mwIndex *ir, mwIndex *jc);
double compute_scalar_gradient_sparse(long lossType, double *x, double *w, double y,
	long d, mwIndex *ir);
void update_test_point_sparse_S2GD(double *x, double *w,
	double sigmoid, double sigmoidold,
	long d, double stepSize, mwIndex *ir);
void lazy_update_S2GD(double *w, double *wold, double *g, long *last_seen,
	double stepSize, double lambda, long i, mwIndex *ir, mwIndex *jc);
void finish_lazy_updates_S2GD(double *w, double *wold, double *g, 
  long *last_seen, double stepSize, double lambda, long iters, long d);

/// Compute the function value of average regularized linear loss
/// lossType - type of loss to be computed
///            1: Quadratic loss
///            2: Logistic loss
/// *w - test point
/// *Xt - data matrix
/// *y - set of labels
/// n - number of training examples
/// d - dimension of the problem
/// lambda - regularization parameter
/// *ir - row indexes of elements of the data matrix
/// *jc - indexes of first elements of columns (size is n+1)
double compute_function_value_sparse(long lossType, double* w, double *Xt, 
        double *y, long n, long d, double lambda, mwIndex *ir, mwIndex *jc)
{
	double value = 0;
	double tmp;
	// Compute losses of individual functions and average them
	for (long i = 0; i < n; i++) {
		tmp = 0;
		for (long j = jc[i]; j < jc[i + 1]; j++) {
			tmp += Xt[j] * w[ir[j]];
		}
    switch (lossType) {
      case 1 : // Quadratic loss
        value += pow(y[i] - tmp, 2) / 2;
        break;
      case 2 : // Logistic Loss
        value += log(1 + exp(y[i] * tmp));
        break;
      case 3: // Hinge squared
        tmp *= y[i];
        if (tmp > 1) {
          // value += 0; (do nothing)
        } else {
          if (tmp < 0) {
            value += 0.5 - tmp;
          } else {
            value += 0.5 * (1 - tmp) * (1 - tmp);
          }
        }
        break;
    default : // Throw error if loss not specified or something weird
  		mexErrMsgTxt("Something gone wrong with the loss specification.");
      break;
    }
  }
	value = value / n;

	// Add regularization term
	for (long j = 0; j < d; j++) {
		value += (lambda / 2) * w[j] * w[j];
	}
	return value;
}

/// Computes the gradient of the entire function,
/// for sparse data matrix. Gradient is changed in place in g. 
/// lossType - type of loss to be computed
///            1: Quadratic loss
///            2: Logistic loss
/// *Xt - sparse data matrix; examples in columns!
/// *w - test point
/// *y - set of labels
/// *g - gradient; updated in place; input value irrelevant
/// n - number of training examples
/// d - dimension of the problem
/// lambda - regularization parameter
/// *ir - row indexes of elements of the data matrix
/// *jc - indexes of first elements of columns (size is n+1)
void compute_full_gradient_sparse(long lossType, double *Xt, double *w, double *y, 
  double *g, long n, long d, double lambda, mwIndex *ir, mwIndex *jc)
{
	// Initialize the gradient
	for (long i = 0; i < d; i++) {
		g[i] = 0;
	}

	// Sum the gradients of individual functions
	double tmp;
	for (long i = 0; i < n; i++) {
		tmp = compute_scalar_gradient_sparse(lossType, Xt + jc[i], w, y[i], 
            jc[i + 1] - jc[i], ir + jc[i]);
		for (long j = jc[i]; j < jc[i + 1]; j++) {
			g[ir[j]] += Xt[j] * tmp;
		}
	}

	// Average the gradients and add gradient of regularizer
	for (long i = 0; i < d; i++) {
		g[i] = g[i] / n;
		g[i] += lambda * w[i];
	}
}

/// compute_sigmoid_sparse computes the derivative of least suqare loss,
/// i.e. i.e. "x^T*w - y_i" for sparse data --- sparse x
/// lossType - type of loss to be computed
///            1: Quadratic loss
///            2: Logistic loss
/// *x - pointer to the first element of the data point
///      (e.g. Xt+jc[i] for i-th example)
/// *w - test point
/// y - label of the training example
/// d - number of nonzeros for current example (*x)
/// *ir - contains row indexes of elements of *x
///		  pointer to the first element of the array 
///		  (e.g. ir+jc[i] for i-th example)
double compute_scalar_gradient_sparse(long lossType, double *x, double *w, 
        double y, long d, mwIndex *ir)
{
	double tmp = 0;
	// Sparse inner product
	for (long j = 0; j < d; j++) {
		tmp += w[ir[j]] * x[j];
	}
  // Apply the loss derivative
  switch (lossType) {
    case 1 : // Quadratic loss
      tmp -= y;
      break;
    case 2 : // Logistic loss
      tmp = exp(y * tmp);
      tmp = y * tmp / (1 + tmp);
      break;
    case 3: // Hinge squared
      tmp *= y;
      if (tmp > 1) {
        tmp = 0; 
      } else {
        if (tmp < 0) {
          tmp = - y;
        } else {
          tmp = - y * (1 - tmp);
        }
      }
      break;
    default : // Throw error if loss not specified or something weird
  		mexErrMsgTxt("Something gone wrong with the loss specification.");
      break;
  }
	return tmp;
}

/// Updates the test point *w in place
/// Makes the step only in the nonzero coordinates of *x,
/// and without regularizer. The regularizer step is constant
/// across more iterations --- updated in lazy_updates
/// *x - training example
/// *w - test point; updated in place
/// gradient - scalar gradient at current point *w
/// gradientold - scalar gradient at old point *wold
/// d - number of nonzeros of training example *x
/// stepSize - stepsize parameter
/// *ir - row indexes of nonzero elements of *x
void update_test_point_sparse_S2GD(double *x, double *w,
	double gradient, double gradientold,
	long d, double stepSize, mwIndex *ir)
{
	for (long j = 0; j < d; j++) {
		w[ir[j]] -= stepSize * (x[j] * (gradient - gradientold));
	}
}


//////////////////////////////////////////////////////////////////////////
///  Lazy updates for SVRG type of algorithms                          ///
//////////////////////////////////////////////////////////////////////////

/// Performs "lazy, in time" update, to obtain current value of 
/// specific coordinates of test point, before a sparse gradient 
/// is to be computed. For SVRG algorithm
/// *w - test point; updated in place
/// *wold - old test point, where full gradient was computed
/// *g - full gradient computed at point *wold
/// *last_seen - numbers of iterations when corresponding 
///				 coordinate was updated last time
/// stepSize - stepsize parameter
/// lambda - regularization paramteter
/// i - number of iteration from which this lazy update was called
/// *ir - row indexes of nonzero elements of training example,
///		  for which the gradient is to be computed
/// *jc - index of element in data matrix where starts the training
///		  exampls for which the gradient is to be computed
void lazy_update_S2GD(double *w, double *wold, double *g, long *last_seen,
	double stepSize, double lambda, long i, mwIndex *ir, mwIndex *jc)
{
	for (long j = *jc; j < *(jc + 1); j++) {
    double c = pow(1 - lambda * stepSize, i - last_seen[ir[j]]);
    w[ir[j]] *= c;
    c = (1 - c) / lambda;
    w[ir[j]] += - c * g[ir[j]] + 
                lambda * c * wold[ir[j]];
		last_seen[ir[j]] = i;
	}
}

/// This is first-order approximation of the update in function 
/// lazy_update_S2GD. It is computationally more efficient, and the 
/// difference from exact update is in most cases negligible, and 
/// diminishes as the algorithm progresses towards optimum.
void lazy_update_S2GD_v2(double *w, double *wold, double *g, long *last_seen,
	double stepSize, double lambda, long i, mwIndex *ir, mwIndex *jc)
{
	for (long j = *jc; j < *(jc + 1); j++) {
		w[ir[j]] -= stepSize * (i - last_seen[ir[j]]) *
			(g[ir[j]] + lambda * (w[ir[j]] - wold[ir[j]]));
		last_seen[ir[j]] = i;
	}
}

/// VERSION FOR CATALYST ACCELERATION
/// Performs "lazy, in time" update, to obtain current value of 
/// specific coordinates of test point, before a sparse gradient 
/// is to be computed. For S2GD algorithm
/// *w - test point; updated in place
/// *wold - old test point, where full gradient was computed
/// *g - full gradient computed at point *wold
/// *last_seen - numbers of iterations when corresponding 
///				 coordinate was updated last time
/// stepSize - stepsize parameter
/// lambda - regularization paramteter
/// i - number of iteration from which this lazy update was called
/// *ir - row indexes of nonzero elements of training example,
///		  for which the gradient is to be computed
/// *jc - index of element in data matrix where starts the training
///		  exampls for which the gradient is to be computed
/// kappa - parameter of the catalyst, regularization strength
/// *yk - parameter of the catalyst, regularization center
void lazy_update_S2GD_acc(double *w, double *wold, double *g, 
  long *last_seen, double stepSize, double lambda, long i, 
  mwIndex *ir, mwIndex *jc, double kappa, double *yk)
{
	for (long j = *jc; j < *(jc + 1); j++) {
    double c = pow(1 - (lambda + kappa) * stepSize, i - last_seen[ir[j]]);
    w[ir[j]] *= c;
    c = (1 - c) / (lambda + kappa);
    w[ir[j]] += - c * g[ir[j]] + 
                lambda * c * wold[ir[j]] + 
                kappa * c * yk[ir[j]]; 
    last_seen[ir[j]] = i;
	}
}

/// This is first-order approximation of the update in function 
/// lazy_update_S2GD_acc. It is computationally more efficient, and the 
/// difference from exact update is in most cases negligible, and 
/// diminishes as the algorithm progresses towards optimum.
void lazy_update_S2GD_acc_v2(double *w, double *wold, double *g, 
  long *last_seen, double stepSize, double lambda, long i, 
  mwIndex *ir, mwIndex *jc, double kappa, double *yk)
{
	for (long j = *jc; j < *(jc + 1); j++) {
		w[ir[j]] -= stepSize * (i - last_seen[ir[j]]) *
			(g[ir[j]] + 
       lambda * (w[ir[j]] - wold[ir[j]]) + 
       kappa * (w[ir[j]] - yk[ir[j]]));
		last_seen[ir[j]] = i;
	}
}

/// Finises the "lazy" updates at the end of outer loop
/// *w - test point; updated in place
/// *wold - old test point, where full gradient was computed
/// *g - full gradient computed at point *wold
/// *last_seen - numbers of iterations when corresponding 
///				 coordinate was updated last time
/// stepSize - stepsize parameter
/// lambda - regularization paramteter
/// iters - number of steps taken in the current outer loop
///			also size of the just finished inner loop
/// d - dimension of the problem
void finish_lazy_updates_S2GD(double *w, double *wold, double *g, long *last_seen,
	double stepSize, double lambda, long iters, long d)
{
	for (long j = 0; j < d; j++) {
    double c = pow(1 - lambda * stepSize, iters - last_seen[j]);
    w[j] *= c;
    c = (1 - c) / lambda;
    w[j] += - c * g[j] + lambda * c * wold[j];
	}
}

/// This is first-order approximation of the update in function 
/// finish_lazy_updates_S2GD. It is computationally more efficient, and the 
/// difference from exact update is in most cases negligible, and 
/// diminishes as the algorithm progresses towards optimum.
void finish_lazy_updates_S2GD_v2(double *w, double *wold, double *g, long *last_seen,
	double stepSize, double lambda, long iters, long d)
{
	for (long j = 0; j < d; j++) {
		w[j] -= stepSize * (iters - last_seen[j]) *
			(g[j] + lambda * (w[j] - wold[j]));
	}
}

/// VERSION FOR CATALYST ACCELERATION
/// Finises the "lazy" updates at the end of outer loop
/// *w - test point; updated in place
/// *wold - old test point, where full gradient was computed
/// *g - full gradient computed at point *wold
/// *last_seen - numbers of iterations when corresponding 
///				 coordinate was updated last time
/// stepSize - stepsize parameter
/// lambda - regularization paramteter
/// iters - number of steps taken in the current outer loop
///			also size of the just finished inner loop
/// d - dimension of the problem
/// kappa - parameter of the catalyst, regularization strength
/// *yk - parameter of the catalyst, regularization center
void finish_lazy_updates_S2GD_acc(double *w, double *wold, double *g, 
  long *last_seen, double stepSize, double lambda, long iters, long d, 
  double kappa, double *yk)
{
	for (long j = 0; j < d; j++) {
    double c = pow(1 - (lambda + kappa) * stepSize, iters - last_seen[j]);
    w[j] *= c;
    c = (1 - c) / (lambda + kappa);
    w[j] += - c * g[j] + 
                lambda * c * wold[j] + 
                kappa * c * yk[j]; 
	}
}

/// This is first-order approximation of the update in function 
/// finish_lazy_updates_S2GD_acc. It is computationally more efficient, and
/// the difference from exact update is in most cases negligible, and 
/// diminishes as the algorithm progresses towards optimum.
void finish_lazy_updates_S2GD_acc_v2(double *w, double *wold, double *g, 
  long *last_seen, double stepSize, double lambda, long iters, long d, 
  double kappa, double *yk)
{
	for (long j = 0; j < d; j++) {
		w[j] -= stepSize * (iters - last_seen[j]) *
			(g[j] + 
       lambda * (w[j] - wold[j]) + 
       kappa * (w[j] - yk[j]));
	}
}

//////////////////////////////////////////////////////////////////////////
/// COCOA stuff below ////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

/// Computes the update value to a sampled dual variable
/// *w - test point; updated in place
/// alpha - dual variable corresponding to *x
/// *x - training example
/// y - target label
/// lambda_n - regularization parameter times n (number of data points)
/// norm - squared norm of *x
/// d - number of nonzero elements of the data point *x
/// *ir - row indexes of nonzero elements of data point *x
double SDCA_step_sparse(long lossType, double *w, double alpha, double *x, double y, 
        double lambda_n, double norm, long d, mwIndex *ir) {
	double tmp = 0;
	// Inner product
	for (long i = 0; i < d; i++) {
		tmp += w[ir[i]] * x[i];
	}
  double result = 0;
  
  double numerator;
  double denominator;
  switch (lossType) {
    case 1 : // Quadratic loss
      numerator = y - tmp - alpha;
      denominator = 1 + norm / lambda_n;
      result = numerator / denominator;
      break;
    case 2 : // Logistic loss
      numerator = -y / (1 + exp(-y * tmp)) - alpha;
      denominator = 0.25 + norm / lambda_n;
      if (denominator < 1) { denominator = 1; }
      result = numerator / denominator;
      break;
    case 3: // Hinge squared
      numerator = 1 - y * tmp - alpha * y;
      denominator = 1 + norm / lambda_n;
      tmp = numerator / denominator + alpha * y;
      if (tmp > 1) { tmp = 1; }
      if (tmp < 0) { tmp = 0; }
      result = y * tmp - alpha;
      break;
    default : // Throw error if loss not specified or something weird
  		mexErrMsgTxt("Something gone wrong with the loss specification.");
      break;
  }

  return result;
}

/// Applies update to primal variable
/// *w - test point; updated in place
/// *x - training example
/// update - computed update to corresponding primal variable
/// d - number of nonzero elements of the data point *x
/// lambda_n - regularization parameter times n (number of data points)
/// *ir - row indexes of nonzero elements of data point *x
void update_primal_sparse(double *w, double *x, double update, long d, 
        double lambda_n, mwIndex *ir) {
  double c = update / lambda_n;
  for (long i = 0; i < d; i++) {
    w[ir[i]] += x[i] * c;
  }
}
