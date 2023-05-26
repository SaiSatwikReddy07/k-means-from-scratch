% Define the number of data points in each group
n = 50;


% 'repmat' stands for "repeat matrix" and 
% allows you to create an array by repeating 
% another array multiple times in a specified pattern.

% 'randn' stands for "random normal" and 
% allows you to generate an array of 
% random numbers with a normal (Gaussian) distribution. 

% Generate the first group of data points
group1 = randn(n,2) + repmat([2 2],n,1);

% Generate the second group of data points
group2 = randn(n,2) + repmat([-2 -2],n,1);

% Generate the third group of data points
group3 = randn(n,2) + repmat([2 -2],n,1);

% Combine the data points into a single matrix
data = [group1; group2; group3];

% Set the number of clusters
K = 3;

%The usage of randperm in data(randperm(size(data,1),K),:) 
% is to randomly select K rows from the data matrix

% Randomly initialize the centroids
centroids = data(randperm(size(data,1),K),:);

% Set the maximum number of iterations
maxIter = 100;

% Initialize the cluster assignments
clusterAssignments = zeros(size(data,1),1);

% Perform K-means clustering
for iter = 1:maxIter
    
    % Assign each data point to the closest centroid
    for i = 1:size(data,1)
        distances = sqrt(sum((data(i,:) - centroids).^2,2));
        [~, clusterAssignments(i)] = min(distances);
    end
    
    % Update the centroids
    for k = 1:K
        centroids(k,:) = mean(data(clusterAssignments == k,:));
    end
    
end

% Plot the results
figure;
scatter(data(clusterAssignments == 1,1),data(clusterAssignments == 1,2),'r');
hold on;
scatter(data(clusterAssignments == 2,1),data(clusterAssignments == 2,2),'g');
scatter(data(clusterAssignments == 3,1),data(clusterAssignments == 3,2),'b');
scatter(centroids(:,1),centroids(:,2),'k','filled');
title('K-means Clustering');
legend('Cluster 1','Cluster 2','Cluster 3','Centroids');


disp("Final Centroids:");
disp(centroids);

