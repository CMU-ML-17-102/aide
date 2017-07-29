function problem = GenerateProblemLinSparse(K, n, d, omega, lambda, sigma)
% GenerateProblemLinSparse - Generates random Least Squares problem for 
% simulated distributed optimizaiton with sparse data
%
% Inputs:
% K - number of nodes
% n - number of data points per node
% d - dimension of the problem
% omega - number of nonzero elements per data point
% lambda - L2 regularization strength
% sigma - (optional) variance of noise added to real output
% 
% Outputs:
% problem - struct describing the optimization problem
  
  % If solution is set to be true, exact solution is computed in closed
  % form. Set to false for high d problems (involves solving dxd system)
  solution = true;
  if nargin < 6
    sigma = 0.1; % Default value
  end

  % problem parameters
  problem = struct();
  problem.nodes = K;
  problem.nPerNode = n;
  problem.nTotal = n * K;
  problem.dim = d;
  problem.regularizer = lambda;
  
  % Data matrices are stored in problem.data, outputs to be predicted are
  % in problem.labels
  problem.data = cell(problem.nodes, 1);
  problem.labels = cell(problem.nodes, 1);
  problem.wreal = randn(problem.dim, 1);
  
  % Generate the random data as a sparse matrix
  for k = 1:problem.nodes
    i = zeros(omega, problem.nPerNode);
    for idx = 1:problem.nPerNode
      i(:, idx) = randsample(problem.dim, omega);
    end
    i = reshape(i, numel(i), 1); % row indices of nonzero elements
    j = repmat(1:problem.nPerNode, omega, 1);
    j = reshape(j, numel(j), 1); % column indices of nonzero elements
    nonz = randn(size(i)); % random values of nonzero elements
    problem.data{k} = sparse(i, j, nonz, problem.dim, problem.nPerNode);
    
    problem.labels{k} = (problem.wreal' * problem.data{k})' ...
                      + randn(problem.nPerNode, 1) * sigma;
    problem.norms{k} = full(sum(problem.data{k}.^2, 1));
  end
  
  if solution
    % Compute the exact solution in closed form
    % problem.wstar stores the optimal point
    % problem.fstar stores the optimal function value
    XtX = zeros(problem.dim, problem.dim);
    Xtb = zeros(problem.dim, 1);
    for k = 1:problem.nodes
      XtX = XtX + problem.data{k} * problem.data{k}';
      Xtb = Xtb + problem.data{k} * problem.labels{k};
    end
    XtX = XtX ./ problem.nTotal;
    Xtb = Xtb ./ problem.nTotal;
    problem.wstar = (XtX + lambda * eye(d)) \ Xtb;
    problem.fstar = FunctionValueLin(problem, problem.wstar, 1);
  else
    % Leave the optimal fields blank
    problem.wstar = [];
    problem.fstar = [];
  end
  
end