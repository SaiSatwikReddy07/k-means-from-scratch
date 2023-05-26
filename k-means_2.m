
%%
% Load the iris dataset

load fisheriris
X = meas;
%iris_data = readmatrix('iris.csv'); 
%X = iris_data(:, 1:4);


% Standardize the data
X = zscore(X);

% Initialize variables
K = 10;
max_iter = 100;
cost = zeros(1,K);

% Compute clustering for each value of K
for k = 1:K
    % Initialize centroids
    centroids = X(randperm(size(X,1),k),:);

    % Perform K-means clustering
    for i = 1:max_iter
        % Compute distances from each point to each centroid
        dists = pdist2(X,centroids);

        % Assign points to nearest centroid
        [~,idx] = min(dists,[],2);

        % Update centroids
        for j = 1:k
            centroids(j,:) = mean(X(idx==j,:));
        end
    end

    % Compute the cost function for this value of K
    cost(k) = sum(min(pdist2(X,centroids),[],2).^2);
end

% Plot the cost function vs K
figure;
plot(1:K,cost,'-o');
xlabel('Number of clusters (K)');
ylabel('Cost');
title('Elbow Method');

% Choose the optimal value of K
optimal_K = elbow(cost);
fprintf('The optimal value of K is %d\n',optimal_K);

% Perform K-means clustering with the optimal value of K
centroids = X(randperm(size(X,1),optimal_K),:);
for i = 1:max_iter
    % Compute distances from each point to each centroid
    dists = pdist2(X,centroids);

    % Assign points to nearest centroid
    [~,idx] = min(dists,[],2);

    % Update centroids
    for j = 1:optimal_K
        centroids(j,:) = mean(X(idx==j,:));
    end
end

% Plot the results
figure;
gscatter(X(:,1),X(:,2),species);
hold on;
plot(centroids(:,1),centroids(:,2),'kx','MarkerSize',15,'LineWidth',3);
xlabel('Sepal length (cm)');
ylabel('Sepal width (cm)');
title(sprintf('K-means clustering with K=%d',optimal_K));

% Define elbow function
%%
function [k_opt] = elbow(cost)
    % Compute the change in cost function
    diff_cost = diff(cost);

    % Compute the second derivative of the cost function
    diff2_cost = diff(diff_cost);

    % Find the elbow point
    k_opt = find(diff2_cost > 0, 1, 'first') + 1;
end