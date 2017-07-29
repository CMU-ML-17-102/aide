function problem = ProcessData(path, K)
% ProcessData - Processes raw data to the problem structure assumed by
% solvers in this code bundle. Randomly partitions the data to specified
% number of nodes.
% 
% Inputs: 
% path - path to the stored dataset - a .mat file. Expect matrix X and 
%        vector y as a output of the load call. y are labels, X are the 
%        data; data points in columns, possibly sparse.
% K - number of nodes to split the data to.
% 
% Outputs:
% problem - struct describing the optimization problem

  load(path);
  
  problem.nodes = K;
  problem.nTotal = size(X, 2);
  problem.nPerNode = floor(problem.nTotal / K);
  problem.dim = size(X, 1);
  problem.regularizer = 1 / problem.nTotal;
  
  % Generate random permutation of indices and split data based on this
  partition = randperm(problem.nTotal);
  for k = 1:problem.nodes
    problem.data{k} = X(:, partition(k:K:end));
    problem.labels{k} = y(partition(k:K:end));
    
    problem.norms{k} = full(sum(problem.data{k}.^2, 1));
  end
  
  problem.wstar = [];
  problem.fstar = [];
  
end