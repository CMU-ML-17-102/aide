function val = FunctionValueLin(problem, w, losstype)
% FunctionValueLin - Computes function value of linear loss function
% 
% Inputs:
% problem - struct describing the optimization problem
% w - point in which the function is to be evaluated
% losstype - integer variable specifying the loss
%            1: Quadratic loss
%            2: Logistic loss
% 
% Outputs: 
% value - computed function value
  
  val = 0;
  
  switch losstype
    case 1 % Quadratic
      for k = 1:problem.nodes
        val = val + norm((w' * problem.data{k})' - problem.labels{k})^2 / 2;
      end
      
    case 2 % Logistic
      for  k = 1:problem.nodes
        val = val + ...
          sum(log(1 + exp((w' * problem.data{k})' .* problem.labels{k})));
      end
      
    case 3 % Squared hinge loss
      for k = 1:problem.nodes
        val = val + sum(HingeSq(...
              (w' * problem.data{k})' .* problem.labels{k}));
      end
      
    otherwise
      error('What loss type did you give me???');
  end
  
  % Our loss is average of losses over training examples
  val = val / problem.nTotal;
  % Add regularization term
  val = val + (problem.regularizer * norm(w)^2 / 2);
  
end

function val = HingeSq(val)
  typea = val > 1;
  typeb = val < 0;
  typec = (val >= 0) & (val <= 1);
  
  val(typea) = 0;
  val(typeb) = 0.5 - val(typeb);
  val(typec) = 0.5 * (1 - val(typec)).^2;
end