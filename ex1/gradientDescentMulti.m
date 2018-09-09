function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = size(X, 1); % number of training examples
n = size(X, 2);
J_history = zeros(num_iters, 1);
temp = zeros(1, n);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
	
	for i = 1:n,
		h = theta' * X';
		sigma = 0;
		for j = 1:m,
			sigma = sigma + (h(j) - y(j)) * X(j, i);
		end
		sigma = (sigma * alpha) / m;
		temp(i) = theta(i) - sigma;
	end
	
	for i = 1:n,
		theta(i) = temp(i);
	end
	
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end
end

