function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

%%hypotesis
hypo = X*theta;

%%regularization value
regParam = lambda * sum(theta(2:end,:).^2);

%%square error
sError =  sum((hypo-y).^2);

%%total cost
J= (1/(2*m)) * (sError + regParam);

%%gradient 

grad(1)=sum((hypo-y).*X(:,1));

numOfTheta = max(size(theta));

for i=2:numOfTheta
    regParamGrad = lambda * sum(theta(i));
    sErrorGrad =  sum((hypo-y).*X(:,i));
    grad(i) = (sErrorGrad+regParamGrad);
end

grad = grad / m;



% =========================================================================

grad = grad(:);

end
