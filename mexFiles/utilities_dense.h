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
//// This contains functions for dense data matrix                ////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

// Declarations
double compute_function_value(long lossType, double* w, double *Xt, 
  double *y, long n, long d, double lambda);
void compute_full_gradient(long lossType, double *Xt, double *w, 
  double *y, double *g, long n, long d, double lambda);
double compute_scalar_gradient(long lossType, double *x, double *w, 
  double y, long d);
void update_test_point_dense(double *w, double *g, long d,double stepSize);
void update_test_point_dense_S2GD(double *x, double *w, double *wold, 
	double *gold, double gradient, double gradientold,
	long d, double stepSize, double lambda);
void update_test_point_dense_S2GD_acc(double *x, double *w, double *wold, 
	double *gold, double gradient, double gradientold,
	long d, double stepSize, double lambda, double kappa, double *yk);


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
double compute_function_value(long lossType, double* w, double *Xt, 
        double *y, long n, long d, double lambda)
{
	double value = 0;
	double tmp;
	// Compute losses of individual functions and average them
	for (long i = 0; i < n; i++) {
		tmp = 0;
    // Inner product
		for (long j = 0; j < d; j++) {
			tmp += Xt[i*d + j] * w[j];
		}
    // The loss function
    switch (lossType) {
      case 1 : // Quadratic loss
        value += pow(y[i] - tmp, 2) / 2;
        break;
      case 2 : // Logistic loss
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

/// compute_full_gradient computes the gradient 
/// of the entire loss function. Gradient is changed in place in *g
/// lossType - type of loss to be computed
///            1: Quadratic loss
///            2: Logistic loss
/// *Xt - data matrix; examples in columns!
/// *w - test point
/// *y - set of labels
/// *g - gradient; updated in place; input value irrelevant
/// n - number of training examples
/// d - dimension of the problem
/// lambda - regularization parameter
void compute_full_gradient(long lossType, double *Xt, double *w, double *y,
        double *g, long n, long d, double lambda)
{
	// Initialize the gradient
	for (long i = 0; i < d; i++) {
		g[i] = 0;
	}

	// Sum the gradients of individual functions
	double tmp;
  for (long i = 0; i < n; i++) {
    tmp = compute_scalar_gradient(lossType, Xt + d*i, w, y[i], d);
    for (long j = 0; j < d; j++) {
			g[j] += Xt[d*i + j] * tmp;
		}
	}

	// Average the gradients and add gradient of regularizer
	for (long i = 0; i < d; i++) {
		g[i] = g[i] / n;
		g[i] += lambda * w[i];
	}
}

/// compute_scalar_gradient computes the scalar derivative of linear model
/// lossType - type of loss to be computed
///            1: Quadratic loss
///            2: Logistic loss
/// *x - pointer to the first element of the training example
///		 e.g. Xt + d*i for i-th example
/// *w - test point
/// y - label of the training example
/// d - dimension of the problem
double compute_scalar_gradient(long lossType, double *x, double *w, 
        double y, long d)
{
	double tmp = 0;
	// Inner product
	for (long j = 0; j < d; j++) {
		tmp += w[j] * x[j];
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

/// Update the test point *w in place
/// Hint: Use this function only when you assume the gradient is fully dense
/// *w - test point; updated in place
/// *g - gradient (update direction)
/// d - dimension of the problem
/// stepSize - step-size parameter
void update_test_point_dense(double *w, double *g, long d, double stepSize)
{
	for (long j = 0; j < d; j++) {
		w[j] -= stepSize * (g[j]);
	}
}

/// Update the test point *w in place once you have everything prepared
/// *x - training example
/// *w - test point; updated in place
/// *wold - old test point, where full gradient was computed
/// *gold - full gradient computed at point *wold
/// gradient - scalar gradient at current point *w
/// gradientold - scalar gradient at old point *wold
/// d - dimension of the problem
/// stepSize - stepsize parameter
/// lambda - regularization paramteter
void update_test_point_dense_S2GD(double *x, double *w, double *wold, 
	double *gold, double gradient, double gradientold,
	long d, double stepSize, double lambda)
{
	for (long j = 0; j < d; j++) {
		w[j] -= stepSize * (gold[j] + x[j] *
			(gradient - gradientold) + lambda * (w[j] - wold[j]));
	}
}

/// Update the test point *w in place once you have everything prepared, 
/// for the version with Universal Catalyst
/// *x - training example
/// *w - test point; updated in place
/// *wold - old test point, where full gradient was computed
/// *gold - full gradient computed at point *wold
/// gradient - scalar gradient at current point *w
/// gradientold - scalar gradient at old point *wold
/// d - dimension of the problem
/// stepSize - stepsize parameter
/// lambda - regularization paramteter
/// kappa - catalyst regularization parameter
/// *yk - center of catalyst regularization
void update_test_point_dense_S2GD_acc(double *x, double *w, double *wold, 
	double *gold, double gradient, double gradientold,
	long d, double stepSize, double lambda, double kappa, double *yk)
{
	for (long j = 0; j < d; j++) {
		w[j] -= stepSize * (gold[j] + x[j] * (gradient - gradientold) + 
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
/// d - dimension of the problem
double SDCA_step(long lossType, double *w, double alpha, double *x, double y, 
        double lambda_n, double norm, long d) {
	double tmp = 0;
	// Inner product
	for (long j = 0; j < d; j++) {
		tmp += w[j] * x[j];
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
      numerator = 1 + y * tmp + alpha * y;
      denominator = 1 + norm / lambda_n;
      tmp = numerator / denominator - alpha * y;
      if (tmp > 1) { tmp = 1; }
      if (tmp < 0) { tmp = 0; }
      result = -y * tmp - alpha;
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
/// d - dimension of the problem
/// lambda_n - regularization parameter times n (number of data points)
void update_primal(double *w, double *x, double update, long d, 
        double lambda_n) {
  double c = update / lambda_n;
  for (long i = 0; i < d; i++) {
    w[i] += x[i] * c;
  }
}
