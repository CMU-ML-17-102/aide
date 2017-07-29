function alpha = FindAccParameter(a, q)
% FindAccParameter - Returns alpha parameter for catalyst acceleration
% procedure. Equation from step 4 in Algorighm 1 of the Universal Catalyst 
% paper of Lin, Mairal and Harchaoui
% 
% Inputs:
% a - alpha value in previous time step
% q - parameter of the catalyst
% 
% Outputs:
% alpha - new alpha value for catalyst

  % Compute roots of the quadratic equaiton and return the one in (0, 1)
  r = roots([1, a^2 - q, -a^2]);
  if (r(1) > 0) && (r(1) < 1)
    alpha = r(1);
  elseif (r(2) > 0) && (r(2) < 1)
    alpha = r(2);
  else
    fprintf('r(1) = %f   r(2) = %f\n', r(1), r(2));
    error('Something went wrong in FindAccParameter');
  end
  
end