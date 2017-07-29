function grad = GradientLin(problem, w, losstype)
% GradientLin - Computes full gradient of linear loss function
% 
% Inputs:
% problem - struct describing the optimization problem
% w - point in which the gradient is to be evaluated
% losstype - integer variable specifying the loss
%            1: Quadratic loss
%            2: Logistic loss
% 
% Outputs: 
% grad - computed gradient

  grad = zeros(problem.dim, 1);
  
  switch losstype
    case 1 % Quadratic
      for k = 1:problem.nodes
        grad = grad + problem.data{k} * ...
                      ((w' * problem.data{k})' - problem.labels{k});
      end
      
    case 2 % Logistic
      for  k = 1:problem.nodes
        temp = exp((w' * problem.data{k})' .* problem.labels{k});
        temp = problem.labels{k} .* temp ./ (1 + temp);
        grad = grad + problem.data{k} * temp;
      end
    
    case 3 % Hinge squared
      for k = 1:problem.nodes
        temp = HingeSqGrad((w' * problem.data{k})' .* problem.labels{k},...
                            problem.labels{k});
        grad = grad + problem.data{k} * temp;
      end
      
    otherwise
      error('What loss type did you give me???');
  end
  
  % Our loss is average of losses over training examples
  grad = grad / problem.nTotal;
  % Add regularization term
  grad = grad + problem.regularizer * w;
  
end

function grad = HingeSqGrad(grad, y)
  typea = grad > 1;
  typeb = grad < 0;
  typec = (grad >= 0) & (grad <= 1);
  
  grad(typea) = 0;
  grad(typeb) = -y(typeb);
  grad(typec) = -y(typec) .* (1 - grad(typec));
end