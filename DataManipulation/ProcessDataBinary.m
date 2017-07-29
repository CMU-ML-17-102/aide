function problem = ProcessDataBinary(path)
% ProcessData - Processes raw data to the problem structure assumed by
% solvers in this code bundle. Partitions the data to two groups - positive
% and negative training examples
% 
% Inputs: 
% path - path to the stored dataset - a .mat file. Expect matrix X and 
%        vector y as a output of the load call. y are labels, X are the 
%        data; data points in columns, possibly sparse.
% 
% Outputs:
% problem - struct describing the optimization problem

  load(path);
  
  problem.nodes = 2;
  problem.nTotal = size(X, 2);
  problem.nPerNode = [];
  problem.dim = size(X, 1);
  problem.regularizer = 1 / problem.nTotal;
  
  % Generate random permutation of indices and split data based on this
  problem.data{1} = X(:, y == 1);
  problem.labels{1} = y(y == 1);
  problem.data{2} = X(:, y == -1);
  problem.labels{2} = y(y == -1);

  problem.norms{1} = full(sum(problem.data{1}.^2, 1));
  problem.norms{2} = full(sum(problem.data{2}.^2, 1));
  
  problem.wstar = [];
  problem.fstar = [];
  
end