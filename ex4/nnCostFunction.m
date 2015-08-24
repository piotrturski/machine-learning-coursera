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

% X = 5000 x 400
Activations = X';
Thetas = {Theta1, Theta2}; %1: 25 x 401; 2: 10 x 26; theta j: j -> j+1
%fprintf('size Theta1: %d x %d\n', size(Theta1));
%fprintf('size Theta2: %d x %d\n', size(Theta2));
Top_ones = ones(1, m);
Answers_enc = eye(num_labels)(:,y); % 10 x 5000; K x m
thetas_squares = 0;
Z = {};
A = {};
layers = length(Thetas) + 1;

% forward propagation: h(x) = Activations
A{1} = Activations;
for i=1:length(Thetas)
  A{i} = [Top_ones ; A{i}];
  z = Thetas{i} * A{i};
  A{i+1} = sigmoid(z);
  Z{i+1} = z;
 % fprintf('size z%d: %dx%d\n', i, size(z));

  thetas_squares += sum (sum (Thetas{i}(:,2:end) .^ 2)); % regularization
  %fprintf('Theta%d size: %dx%d\n', i, size(Thetas{i}));
end

Output = A{length(Thetas)+1};
J = sum(sum(-Answers_enc .* log(Output) - (1 - Answers_enc) .* log(1 - Output) )) / m ...
        + lambda / (2 * m) * thetas_squares;

% backward propagation
d = {}; % error of node from incoming? signal
d{layers} = Output - Answers_enc;
for i = layers-1:-1:2
  d{i} = ((Thetas{i}(:,2:end))' * d{i+1}) .* sigmoidGradient(Z{i});
end

%disp('A: '); disp(size(A));
%fprintf('number of layers: %d\n', layers);
D = {}; % same as theta; j -> j+1
for i = 1: layers - 1
  %fprintf('D for layer %d; d{%d}: %d x %d; A{%d}: %d x %d\n', i, i+1, size(d{i+1}), i, size(A{i}) );
  Thetas{i}(:,1) = 0; % we won't need theta anymore so change it for easy regularization
  D{i} = d{i+1} * A{i}' / m + lambda * Thetas{i} / m;
  % fprintf('D for layer %d computed. size: %d x %d\n', i, size(D{i}));
end

%fprintf('size D: %d x %d\n', size(D));
Theta1_grad = D{1};
Theta2_grad = D{2};

%fprintf('th1_grad: %d x %d\n', size(Theta1_grad));
%fprintf('th2_grad: %d x %d\n', size(Theta2_grad));

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
