function [w, hist] = InexactDANELinSVRG(problem, w, h, iters, passes, ...
                                        losstype)
% InexactDANELinSVRG - Inexact DANE algorithm for linear problem. Uses SVRG
% algorighm as local solver to obtain approximate solution of the DANE
% subproblem.
% 
% Inputs:
% problem - struct describing the optimization problem
% w - starting point of the optimization
% h - stepsize to be used for the local SVRG solver
% iters - number of iterations to run the algorithm for
% passes - number of passes over local data local SVRG should run for
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
  else
    computeFunctionValues = false;
  end
  
  % Iterations of the algotihm
  for i = 1:iters
    % Initialize new iterate and compute gradient
    wnew = zeros(size(w));
    grad = GradientLin(problem, w, losstype);
    
    % Run local algorithms
    for k = 1:problem.nodes
      wstart = w; wstart(1) = 0; wstart(1) = w(1); % MAGIC! DO NOT TOUCH!!!
      % (or ask Jakub for meaning and do not touch then)
      
      % Determine how many how big local iterations to do. m is single
      % value as we run SVRG for a single outer loop. iVals is sequence of
      % indices of sampled data points
      n = length(problem.labels{k});
      m = int64(round(n * passes));
      
      pass = length(problem.labels{k}); s = 1:pass;
      iVals = int64(zeros(round(length(s) * passes), 1));
      % Sample the iterations. 
      for j = 1:ceil(passes)
        % Do random permutation of local data
        s = s(randperm(length(s)))';
        if passes - j < 0
          iVals(((j-1)*pass + 1):end) = ...
                  s(1:length(iVals(((j-1)*pass + 1):end)));
        else
          iVals(((j-1)*pass + 1):(j*pass)) = s;          
        end
      end
      iVals = iVals - 1; % C++ indexes from 0
      
      % Call the SVRG on local data
      SVRGLocal(wstart, problem.data{k}, problem.labels{k}, ...
                 problem.regularizer, grad, h, iVals, m, losstype);
      
      wnew = wnew + wstart;
    end
    % Average the local updates
    w = wnew / problem.nodes;
    
    if computeFunctionValues
      hist(i + 1) = FunctionValueLin(problem, w, losstype);
    end
  end
  
end