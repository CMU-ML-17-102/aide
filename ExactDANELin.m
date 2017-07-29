function [w, hist] = ExactDANELin(problem, w, iters, losstype)
% ExactDANELin - DANE algorithm for linear problem. Runs the algorithm
% presented in DANE paper by Shamir, Srebro and Zhang.
% 
% Inputs:
% problem - struct describing the optimization problem
% w - starting point of the optimization
% iters - number of iterations to run the algorithm for
% losstype - integer variable specifying the loss
%            1: Quadratic loss
%            2: Logistic loss
% 
% Outputs: 
% w - resulting point of the optimization procedure
% hist - history of function values during the run of algorithm

  if nargout > 1
    % Compute the function values only if requested as output
    computeFunctionValues = true;
    hist = zeros(iters + 1, 1);
    hist(1) = FunctionValueLin(problem, w, losstype);
  end
  
  for i = 1:iters
    wnew = zeros(size(w));
    % Compute gradient
    grad = GradientLin(problem, w, losstype);
    
    for k = 1:problem.nodes
      nLocal = size(problem.data{k}, 2);
      % Do DANE step. Eq (16) in the DANE paper...
      wnew = wnew + (w - ((1 / nLocal) * ...
                            problem.data{k} * problem.data{k}') \ grad);
    end
    
    % Average the resulting iterates
    w = wnew / problem.nodes;
    if computeFunctionValues
      hist(i + 1) = FunctionValueLin(problem, w, losstype);
    end
  end
  
end