function [w, alpha, hist] = CocoaSDCA(problem, w, alpha, nu, iters, ...
                                      passes, losstype)
% CocoaSDCA - Cocoa algorithm with SDCA as the local solver
% 
% Inputs:
% problem - struct describing the optimization problem
% w - starting primal point of the optimization
% alpha - starting dual point of the optimization. cell array of size
%         problem.nodes; dual variables partitioned with data
% iters - number of iterations to run the algorithm for
% passes - number of passes over local data local SVRG should run for
% 
% Outputs: 
% w - resulting primal point of the optimization procedure
% alpha - resulting dual point of the optimization procedure
% hist - history of function values during the run of algorithm
  
  % If empty primal variable is provided, initialize to zero.
  if isempty(w)
    w = zeros(problem.dim, 1);
    for k = 1:problem.nodes
      alpha{k} = zeros(length(problem.labels{k}), 1);
    end
  end
  
  if nargout > 2
    % Compute the function values only if requested as output
    computeFunctionValues = true;
    hist = zeros(iters + 1, 1);
    hist(1) = FunctionValueLin(problem, w, losstype);
  else
    computeFunctionValues = false;
  end
  
  alphaupdate = cell(problem.nodes, 1);
  % Iterations of the algorithm
  for i = 1:iters
    
    for k = i:problem.nodes
      alphaupdate{k} = zeros(size(alpha{k}));
    end
    
    % Local iterations
    for k = 1:problem.nodes
      pass = length(problem.labels{k}); s = 1:pass;
      iVals = int64(zeros(length(s) * passes, 1));
      % Sample the iterations. 
      for j = 1:passes
        % Do random permutation of local data
        s = s(randperm(length(s)))';
        iVals(((j-1)*pass + 1):(j*pass)) = s;
      end
      iVals = iVals - 1; % C++ indexes from 0
      
      alphastart = alpha{k}; alphastart(1) = 0; alphastart(1) = alpha{k}(1);
      wstart = w; wstart(1) = 0; wstart(1) = w(1); % MAGIC! DO NOT TOUCH!!!
      % (or ask Jakub for meaning and do not touch then)
      SDCALocal(wstart, alphastart, problem.data{k}, problem.labels{k}, ...
        problem.norms{k}, problem.regularizer, iVals, problem.nTotal, losstype);        
      
      alphaupdate{k} = alphastart - alpha{k};
    end
    
    % Form new dual iterate
    for k = 1:problem.nodes
      alpha{k} = alpha{k} + nu * alphaupdate{k};
    end
    
    % Form the new primal iterate (recompute from scratch)
    w = zeros(size(w));
    for k = 1:problem.nodes
      w = w + problem.data{k} * alpha{k};
    end
    w = w ./ (problem.regularizer * problem.nTotal);

    if computeFunctionValues
      hist(i + 1) = FunctionValueLin(problem, w, losstype);
    end

  end
  
end