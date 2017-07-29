function [w, hist] = AIDELinSVRG(problem, w, h, kappa, q, iters,...
                                 passes, losstype)
  
% AIDELinSVRG - Accelerated Inexact DANE algorithm (AIDE) for linear 
% problem. Uses SVRG algorighm as local solver to obtain approximate 
% solution of the DANE subproblem. Acceleration is done in the sense of
% Universal Catalyst of Lin, Mairal and Harchaoui
% 
% Inputs:
% problem - struct describing the optimization problem
% w - starting point of the optimization
% h - stepsize to be used for the local SVRG solver
% kappa - parameter of catalyst, additional regularization strength
% q - parameter of catalyst, usually function of problem.regularizer and
%     kappa; not necessarily though
% iters - number of iterations to run the algorithm for
% passes - number of passes over local data local SVRG should run for
% losstype - integer variable specifying the loss
%            1: Quadratic loss
%            2: Logistic loss
% 
% Outputs: 
% w - resulting point of the optimization procedure
% hist - history of function values during the run of algorithm

  % Initialize alpha sequence of catalyst
  alpha = FindAccParameter(1, q);
  if nargout > 1
    % Compute the function values only if requested as output
    computeFunctionValues = true;
    hist = zeros(iters + 1, 1);
    hist(1) = FunctionValueLin(problem, w, losstype);
  else
    computeFunctionValues = false;
  end
  y = w; % initial acceleration point
  
  % Iterations of the algotihm
  for i = 1:iters
    % Initialize new iterate and compute gradient
    wnew = zeros(size(w));
    grad = GradientLin(problem, w, losstype); 
    % !!! Note that here is the gradient without the additional term from
    % accelerated regularization, as that cancels out in the implementation
    % of SVRG for this.
    
    % Run local algorithms
    for k = 1:problem.nodes
      wstart = w; wstart(1) = 0; wstart(1) = w(1); % MAGIC! DO NOT TOUCH!!!
      % (or ask Jakub for meaning and do not touch then)
      
      % Determine how many how big local iterations to do. m is single
      % value as we run SVRG for a single outer loop. iVals is sequence of
      % indices of sampled data points
      n = length(problem.labels{k});
      m = int64(n * passes); 
      iVals = int64(floor(n * rand(sum(m),1)));
      % Call the SVRG on local data
      SVRGLocalAcc(wstart, problem.data{k}, problem.labels{k}, ...
               problem.regularizer, grad, h, iVals, m, kappa, y, losstype);
      
      wnew = wnew + wstart;
    end
    % Average the local updates
    wnew = wnew / problem.nodes;
    
    % Update alpha and beta parameters of the catalyst
    alphanew = FindAccParameter(alpha, q);
    beta = (alpha * (1 - alpha)) / (alpha^2 + alphanew);
    alpha = alphanew;
    % Compute the next acceleration point
    y = (1 + beta) * wnew - beta * w;
    % And set the new iterate
    w = wnew;
    
    if computeFunctionValues
      hist(i + 1) = FunctionValueLin(problem, w, losstype);
    end
    
  end
  
end