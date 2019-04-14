function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

A = [ones(m, 1) X];   %5000*401
J_temp = zeros(m, 1); %5000*1

%Forward Propagation
for i = 1:m
    y_vec = zeros(num_labels, 1);        %10*1
    y_vec(y(i)) = 1;        %10*1
    A_1 = A(i,:)';          %401*1
    Z_2 = Theta1 * A_1;     %25*401 * 401*1
    A_2 = sigmoid(Z_2);     %25*1
    A_2 = [ones(1 , size(A_2, 2)) ; A_2];%26*1
    Z_3 = Theta2 * A_2;     %10*26 * 26*1
    A_3 = sigmoid(Z_3);     %10*1
    %hyphesis
    h_theta_x = A_3;        %10*1
    J_temp(i) = -sum(y_vec .* log(h_theta_x) + (1 - y_vec) .* log(1 - h_theta_x)); %sum(10*1 .*10*1)
end
%Cost Function
J = (1 / m) * sum(J_temp); 
J = J + (lambda / (2 * m)) * (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));

%Back Propagation
Delta1 = zeros(hidden_layer_size, input_layer_size+1); %25*401
Delta2 = zeros(num_labels,hidden_layer_size+1);        %10*26
for i = 1:m
    y_vec = zeros(num_labels, 1);        %10*1
    y_vec(y(i)) = 1;        %10*1
    A_1 = A(i,:)';          %401*1
    Z_2 = Theta1 * A_1;     %25*401 * 401*1
    A_2 = sigmoid(Z_2);     %25*1
    A_2 = [ones(1 , size(A_2, 2)) ; A_2];%26*1
    Z_3 = Theta2 * A_2;     %10*26 * 26*1
    A_3 = sigmoid(Z_3);     %10*1
    delta3 = A_3 - y_vec;   %10*1
    gz2 = [0; sigmoidGradient(Z_2)]; %26*1
    delta2 = (Theta2)'*delta3 .* gz2; %26*10 * 10*1 .* 26*1
    delta2 = delta2(2:end); %25*1
    Delta2 = Delta2 + delta3 * A_2'; %10*26 = 10*1 * 1*26 
    Delta1 = Delta1 + delta2 * A_1'; %25*401 = 25*1 * 1*401
end
Theta1_grad = (1 / m) * Delta1;
Theta2_grad = (1 / m) * Delta2;

%Regularized Neural Networks
Theta1(:,1) = 0;
Theta1_grad = Theta1_grad + (lambda / m) * Theta1;
Theta2(:,1) = 0;
Theta2_grad = Theta2_grad + (lambda / m) * Theta2;
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end