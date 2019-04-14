function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%
%�����м�����ĳ�ʼ��
A_1 = zeros(size(X,1), size(X,2)+1);
%Z_2 = zeros(m, size(Theta1, 1) + 1);
%A_2 = zeros(m, size(Theta1, 1) + 1);
Z_3 = zeros(m, num_labels);
A_3 = zeros(m, num_labels);
maxp = zeros(size(X, 1), 1);

%A_1(:,1) = 1;
%A_1(:,2:size(X,2)+1) = X;
A_1 = [ones(m, 1) X];
%Z_2(:,1) = 1;
%Z_2(:,2:size(Theta1, 1) + 1) = A_1 * Theta1';
A_2 = sigmoid(A_1 * Theta1');
A_2 = [ones(size(A_2, 1), 1) A_2];
Z_3 = A_2 * Theta2';
A_3 = sigmoid(Z_3);
[maxp, p] = max(A_3, [], 2);

%�����и����⣬���վ�������һ�����������гɹ������ύʧ��
%Ȼ��ʡ�Բ��ֹ��̣������ز��������ú�ֱ�Ӽ���ƫ���У��ύ�ɹ�
%ƫ����û�в��������ǰ�ľ�������

end
