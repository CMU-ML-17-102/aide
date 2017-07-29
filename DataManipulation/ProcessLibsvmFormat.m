function [y, X] = ProcessLibsvmFormat(datapath, outpath)
% ProcessLibsvmFormat - reads data from raw libsvm format, processes the
% data and stores as .mat file for further usage
%
% Inputs:
% datapath - path to the raw data
% outpath - path where the result is to be saved

  % Read the data from libsvm format
  [y, X] = libsvmread(datapath);
  
  % Remove all-zero data points
  norms = sum(X.^2, 2);
  X = X (norms ~= 0, :);
  y = y(norms ~= 0);
  
  % Normalize data so that average norm is equal to 1
  norms = sum(X.^2, 2);
  X = X ./ sqrt(mean(norms));
  
  % For covtype dataset only
  y(y == 2) = -1;
  
  % Add a regularization term
  X = [ones(length(y), 1), X];
  % Transpose the matrix. Since we sample a lot of data points, we want
  % them to be columns, for the sparse case
  X = X';
  
  % save the resulting data matrix and labels
  save(outpath, 'y', 'X', '-v7.3');
  
end