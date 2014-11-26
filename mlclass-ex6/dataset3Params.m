function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

opt_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

lowest_error = bitmax;

for i = 1:length(opt_values)
  for j = 1:length(opt_values)

    model = svmTrain(X, y, opt_values(i), @(x1, x2) gaussianKernel(x1, x2, opt_values(j))); 

    predictions = svmPredict(model, Xval);

    error_val = mean(double(predictions ~= yval));
    fprintf('Error: %f, C: %f, sigma: %f\n', lowest_error, opt_values(i), opt_values(j));

    if error_val < lowest_error
      lowest_error = error_val;
      C = opt_values(i);
      sigma = opt_values(j);
    end
  end
end

fprintf('Lowest error: %f, optimal C: %f, optimal sigma: %f\n', ...
  lowest_error, C, sigma);

% =========================================================================

end
