function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%

e1 = find(idx == 1);
l1 = length(e1);
e2 = find(idx == 2);
l2 = length(e2);
e3 = find(idx == 3);
l3 = length(e3);


for j = 1:l1
    centroids(1, :) = centroids(1, :) + X(e1(j),:);
end
centroids(1, :) = (1 / l1) * centroids(1, :);

for j = 1:l2
    centroids(2, :) = centroids(2, :) + X(e2(j),:);
end
centroids(2, :) = (1 / l2) * centroids(2, :);

for j = 1:l3
    centroids(3, :) = centroids(3, :) + X(e3(j),:);
end
centroids(3, :) = (1 / l3) * centroids(3, :);





% =============================================================


end

