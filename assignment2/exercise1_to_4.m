clear all
close all
clc

rng(1)

dir_dataset = '../datasets/cifar-10-batches-mat/';
dir_result_pics = 'result_pics/';
dir_result_searches = 'result_searches/';


%% Load data

fprintf('Loading data...\n\n');

% training set
TrainingSet = struct('X', [], 'Y', [], 'y', []);
for batch = 1:5
    filename = sprintf('%sdata_batch_%d.mat', dir_dataset, batch);
    [X, Y, y] = LoadBatch(filename);
    TrainingSet.X = [TrainingSet.X X];
    TrainingSet.Y = [TrainingSet.Y Y];
    TrainingSet.y = [TrainingSet.y y];
end

% test set
[X, Y, y] = LoadBatch([dir_dataset 'test_batch.mat']);
TestSet = struct('X', X, 'Y', Y, 'y', y);

% show images
I = reshape(TrainingSet.X, 32, 32, 3, size(TrainingSet.X, 2));
I = permute(I, [2, 1, 3, 4]);
montage(I(:, :, :, 1:500), 'Size', [5,5]);


%% Pre-process data

mean_train = mean(TrainingSet.X, 2);
std_train = std(TrainingSet.X, 0, 2);

% z-score normalization
TrainingSet.X = (TrainingSet.X - mean_train) ./ std_train;
TestSet.X = (TestSet.X - mean_train) ./ std_train;


%% Network architecture

nLayers = 2;    % number of layers
m = 50;         % number of hidden nodes
K = size(TrainingSet.Y, 1);
d = size(TrainingSet.X, 1);
layerSizes = [d, m, K];
n_batch = 100;


%% Check gradients

disp('Checking gradients...');

n_batch_copy = 20;
n_features = 20;
lambda = 1;

trainX = TrainingSet.X(1:n_features, 1:n_batch_copy);
trainY = TrainingSet.Y(:, 1:n_batch_copy);
[W, b] = InitParameters([n_features, m, K]);

[P, H] = EvaluateClassifier(trainX, W, b);
[grad_W, grad_b] = ComputeGradients(trainX, trainY, P, H, W, lambda);
% [ngrad_b, ngrad_W] = ComputeGradsNum(trainX, trainY, W, b, lambda, 1e-5);
[ngrad_b, ngrad_W] = ComputeGradsNumSlow(trainX, trainY, W, b, lambda, 1e-5);

for l = 1:nLayers
    fprintf('Max absolute error grad_W%d: %e\n', l, max(abs(grad_W{l} - ngrad_W{l}), [], 'all'));
    fprintf('Max relative error grad_W%d: %e\n', l, max(abs(grad_W{l} - ngrad_W{l}) ./ max(eps, abs(grad_W{l}) + abs(ngrad_W{l})), [], 'all'));
    fprintf('Max absolute error grad_b%d: %e\n', l, max(abs(grad_b{l} - ngrad_b{l}), [], 'all'));
    fprintf('Max relative error grad_b%d: %e\n', l, max(abs(grad_b{l} - ngrad_b{l}) ./ max(eps, abs(grad_b{l}) + abs(ngrad_b{l})), [], 'all'));
end
fprintf('\n');


%% Check mini-batch GD with cycling learning rate

disp('Checking cycling learning rate...');

% training set: data_batch_1, validation set: data_batch_2
TrainingSet_ = struct('X', TrainingSet.X(:, 1:10000), 'Y', TrainingSet.Y(:, 1:10000), 'y', TrainingSet.y(1:10000));
ValidationSet = struct('X', TrainingSet.X(:, 10000:20000), 'Y', TrainingSet.Y(:, 10000:20000), 'y', TrainingSet.y(10000:20000));

% hyper-parameters
GDparams = [
    struct('n_batch', n_batch, 'lambda', .01, 'eta_min', 1e-5, 'eta_max', 1e-1, 'n_s', 500, 'n_cycles', 1);
    struct('n_batch', n_batch, 'lambda', .01, 'eta_min', 1e-5, 'eta_max', 1e-1, 'n_s', 800, 'n_cycles', 3);
];

[W, b] = InitParameters(layerSizes);
for config = 1:length(GDparams)
    % train
    [Wstar, bstar, f_cost, f_acc, f_eta] = MiniBatchGD(TrainingSet_, ValidationSet, GDparams(config), W, b);
    
    % save figures
    saveas(f_cost, [dir_result_pics sprintf('loss_cost_%d.jpg', config)]);
    saveas(f_acc, [dir_result_pics sprintf('accuracy_%d.jpg', config)]);
    saveas(f_eta, [dir_result_pics sprintf('cycling_learning_rates_%d.jpg', config)]);
    
    % test
    acc = ComputeAccuracy(TestSet.X, TestSet.y, Wstar, bstar);
    fprintf('Test accuracy %d: %.2f%%\n', config, acc*100);
end
fprintf('\n');


%% Coarse-to-fine random search for lambda

disp('Random searching...');

filename = 'random_search.txt';

% validation set: 5000 images, training set: all the rest
[TrainingSet_, ValidationSet] = SplitData(TrainingSet, .1);
n = size(TrainingSet_.X, 2);

% random search
n_trials = 10;
l_min = -4;
l_max = -2;
l = l_min + (l_max - l_min) * rand(1, n_trials);    % random search (prefer this!)
% l = linspace(l_min, l_max, n_trials);               % grid search
l = 10.^l;                                          % log scale

% hyper-parameters
n_cycles = 3;
n_s = 2 * floor(n/n_batch);
GDparams = struct('n_batch', n_batch, 'eta_min', 1e-5, 'eta_max', 1e-1, 'n_s', n_s, 'n_cycles', n_cycles);

% train and save results
fileID = fopen([dir_result_searches filename], 'a');
fprintf(fileID, 'l_min=%d, l_max=%d, n_trials=%d, n_cycles=%d\n\n', l_min, l_max, n_trials, n_cycles);
[W, b] = InitParameters(layerSizes);
for lambda = l
    % train
    fprintf('lambda=%f - training... ', lambda);
    GDparams.lambda = lambda;
    [Wstar, bstar] = MiniBatchGD(TrainingSet_, ValidationSet, GDparams, W, b);
    
    % test and save
    acc = ComputeAccuracy(ValidationSet.X, ValidationSet.y, Wstar, bstar);
    fprintf(fileID, 'lambda=%f - validation accuracy %.2f%%\n', lambda, acc*100);
    fprintf('done\n');
end
fprintf(fileID, '\n\n\n');
fclose(fileID);
fprintf('\n');
close all


%% Train best network

disp('Training best network...');

% validation set: 1000 images, training set: all the rest
[TrainingSet_, ValidationSet] = SplitData(TrainingSet, .02);
n = size(TrainingSet_.X, 2);

% hyper-parameters
GDparams = struct('lambda', 0.002584, 'n_batch', n_batch, 'eta_min', 1e-5, 'eta_max', 1e-1, 'n_s', 2 * floor(n/n_batch), 'n_cycles', 3);

% train
[W, b] = InitParameters(layerSizes);
[Wstar, bstar, f_cost, f_acc, f_eta] = MiniBatchGD(TrainingSet_, ValidationSet, GDparams, W, b);

% save figures
saveas(f_cost, [dir_result_pics 'loss_cost_best.jpg']);
saveas(f_acc, [dir_result_pics 'accuracy_best.jpg']);
saveas(f_eta, [dir_result_pics 'cycling_learning_rates_best.jpg']);

% test
acc = ComputeAccuracy(TestSet.X, TestSet.y, Wstar, bstar);
fprintf('Test accuracy: %.2f%%\n', acc*100);


%% Functions

function [X, Y, y] = LoadBatch(filename)
    A = load(filename);
    X = double(A.data') / 255;  % between 0 and 1
    y = A.labels' + 1;          % between 1 and 10
    K = 10;
    N = size(X, 2);
    Y = zeros(K, N);
    for n = 1:N
        Y(y(n), n) = 1;
    end
end

function [TrainingSet, TestSet] = SplitData(Dataset, percentage)
    K = size(Dataset.Y, 1);
    N = size(Dataset.X, 2);

    TestSet = struct('X', [], 'Y', [], 'y', []);
    TrainingSet = Dataset;
    
    for class = 1:K
        % select randomly percentage of images of this class
        idx = find(TrainingSet.y == class);
        idx = idx(randperm(length(idx)));
        idx = idx(1:round(percentage*N/K));
        
        % add to test set and remove from training set
        TestSet.X = [TestSet.X TrainingSet.X(:, idx)];
        TestSet.Y = [TestSet.Y TrainingSet.Y(:, idx)];
        TestSet.y = [TestSet.y TrainingSet.y(idx)];
        TrainingSet.X(:, idx) = [];
        TrainingSet.Y(:, idx) = [];
        TrainingSet.y(idx) = [];
    end
end

function [W, b] = InitParameters(layerSizes)
    nLayers = length(layerSizes)-1;
    W = cell(nLayers, 1);
    b = cell(nLayers, 1);
    
    for l = 1:nLayers
        nIn = layerSizes(l);
        nOut = layerSizes(l+1);
        
        % weights: Xavier initialization
        std = 1 / sqrt(nIn);
        W{l} = std * randn(nOut, nIn);
        
        % biases: 0
        b{l} = zeros(nOut, 1);
    end
end

function [P, H] = EvaluateClassifier(X, W, b)
    nLayers = length(W);
    H = cell(nLayers, 1);
    H{1} = X;       % add X (for convenience in the computation)
    
    for l = 1:nLayers
        S = W{l}*H{l} + b{l};
        
        if l < nLayers      % hidden layer: ReLU
            H{l+1} = max(0, S);
        else                % output layer: softmax
            P = exp(S) ./ sum(exp(S));
        end
    end
    
    H = H(2:end);   % remove X
end

function J = ComputeCost(X, Y, W, b, lambda)
    P = EvaluateClassifier(X, W, b);
    
    % loss for each image
    N = size(X, 2);
    loss = zeros(1, N);
    for n = 1:N
        loss(n) = -log(Y(:, n)' * P(:, n));
    end
    
    % regularization
    r = 0;
    for l = 1:length(W)
        r = r + sum(W{l}.^2, 'all');
    end
    
    % cost
    J = 1 / length(loss) * sum(loss) + lambda * r;
end

function acc = ComputeAccuracy(X, y, W, b)
    P = EvaluateClassifier(X, W, b);
    [~, ypred] = max(P);
    nCorrect = length(find(ypred == y));
    nTot = size(X, 2);
    acc = nCorrect / nTot;
end

function [grad_W, grad_b] = ComputeGradients(X, Y, P, H, W, lambda)
    n = size(X, 2);
    nLayers = length(W);
    grad_W = cell(nLayers, 1);
    grad_b = cell(nLayers, 1);
    G = P - Y;
    
    % add X to H for convenience in the computation, the formulae contain l instead of l-1 of the slides
    H(2:length(H)+1) = H(:);
    H{1} = X;
    
    for l = nLayers:-1:1
        grad_W{l} = 1 / n * G * H{l}' + 2 * lambda * W{l};
        grad_b{l} = 1 / n * sum(G, 2);
        
        % not the first layer => back-propagate gradient
        if l > 1
            G = W{l}' * G;
            G(H{l} <= 0) = 0;   % same as multiplying by Ind(H(l)>0)
        end
    end
end

function [Wstar, bstar, f_cost, f_acc, f_eta] = MiniBatchGD(TrainingSet, ValidationSet, GDparams, W, b)
    n = size(TrainingSet.X, 2);
    nLayers = length(W);
    
    % get hyper-parameters
    n_batch = GDparams.n_batch;
    lambda = GDparams.lambda;
    eta_min = GDparams.eta_min;
    eta_max = GDparams.eta_max;
    n_s = GDparams.n_s;
    n_cycles = GDparams.n_cycles;
    n_updates = 2 * n_s * n_cycles;
    updates_per_epoch = floor(n / n_batch);

    % stats
    measures_per_cycle = 9;
    updates_per_measure = floor(2 * n_s / measures_per_cycle);
    n_measures = n_cycles * measures_per_cycle;
    measured_updates = [0, zeros(1, n_measures)];
    costs_train = [ComputeCost(TrainingSet.X, TrainingSet.Y, W, b, lambda), zeros(1, n_measures)];
    costs_val = [ComputeCost(ValidationSet.X, ValidationSet.Y, W, b, lambda), zeros(1, n_measures)];
    losses_train = [ComputeCost(TrainingSet.X, TrainingSet.Y, W, b, 0), zeros(1, n_measures)];
    losses_val = [ComputeCost(ValidationSet.X, ValidationSet.Y, W, b, 0), zeros(1, n_measures)];
    acc_train = [ComputeAccuracy(TrainingSet.X, TrainingSet.y, W, b), zeros(1, n_measures)];
    acc_val = [ComputeAccuracy(ValidationSet.X, ValidationSet.y, W, b), zeros(1, n_measures)];
    idx_measure = 2;
    learning_rates = zeros(1, n_updates);
    
    % init result
    Wstar = W(:);
    bstar = b(:);
    
    for t = 1:n_updates
        batch = mod(t-1, updates_per_epoch) + 1;

        % select minibatch
        idx_start = (batch-1) * n_batch + 1;
        idx_end = batch * n_batch;
        idx = idx_start:idx_end;
        batchX = TrainingSet.X(:, idx);
        batchY = TrainingSet.Y(:, idx);

        % forward pass
        [P, H] = EvaluateClassifier(batchX, Wstar, bstar);

        % backward pass
        [grad_W, grad_b] = ComputeGradients(batchX, batchY, P, H, Wstar, lambda);

        % cyclical learning rate
        half_cycle = floor(t / n_s);
        if mod(half_cycle, 2) == 0
            eta = eta_min + (t - half_cycle*n_s) / n_s * (eta_max - eta_min);
        else
            eta = eta_max - (t - half_cycle*n_s) / n_s * (eta_max - eta_min);
        end

        % update
        for l = 1:nLayers
            Wstar{l} = Wstar{l} - eta * grad_W{l};
            bstar{l} = bstar{l} - eta * grad_b{l};
        end

        % stats
        if mod(t, updates_per_measure) == 0
            measured_updates(idx_measure) = t;
            costs_train(idx_measure) = ComputeCost(TrainingSet.X, TrainingSet.Y, Wstar, bstar, lambda);
            costs_val(idx_measure) = ComputeCost(ValidationSet.X, ValidationSet.Y, Wstar, bstar, lambda);
            losses_train(idx_measure) = ComputeCost(TrainingSet.X, TrainingSet.Y, Wstar, bstar, 0);
            losses_val(idx_measure) = ComputeCost(ValidationSet.X, ValidationSet.Y, Wstar, bstar, 0);
            acc_train(idx_measure) = ComputeAccuracy(TrainingSet.X, TrainingSet.y, Wstar, bstar);
            acc_val(idx_measure) = ComputeAccuracy(ValidationSet.X, ValidationSet.y, Wstar, bstar);
            idx_measure = idx_measure+1;
        end
        learning_rates(t) = eta;
    end
    
    % plot cost/loss curve
    f_cost = figure();
    hold on
    plot(measured_updates, costs_train, 'linewidth', 2);
    plot(measured_updates, costs_val, 'linewidth', 2);
    plot(measured_updates, losses_train, '--', 'linewidth', 1.5);
    plot(measured_updates, losses_val, '--', 'linewidth', 1.5);
    xlabel('update step');
    ylabel('cost / loss');
    legend('training cost', 'validation cost', 'training loss', 'validation loss');
    
    % plot accuracy curve
    f_acc = figure();
    hold on
    plot(measured_updates, acc_train, 'linewidth', 2);
    plot(measured_updates, acc_val, 'linewidth', 2);
    xlabel('update step');
    ylabel('accuracy');
    legend('training', 'validation');
    
    % plot cycling learning rates
    f_eta = figure();
    plot(1:length(learning_rates), learning_rates);
    xlabel('update step');
    ylabel('learning rate');
end


%% Provided functions for numerical computation of gradients

function [grad_b, grad_W] = ComputeGradsNum(X, Y, W, b, lambda, h)

    grad_W = cell(numel(W), 1);
    grad_b = cell(numel(b), 1);

    c = ComputeCost(X, Y, W, b, lambda);

    for j=1:length(b)
        grad_b{j} = zeros(size(b{j}));

        for i=1:length(b{j})
            b_try = b;
            b_try{j}(i) = b_try{j}(i) + h;
            c2 = ComputeCost(X, Y, W, b_try, lambda);
            grad_b{j}(i) = (c2-c) / h;
        end
    end

    for j=1:length(W)
        grad_W{j} = zeros(size(W{j}));

        for i=1:numel(W{j})   
            W_try = W;
            W_try{j}(i) = W_try{j}(i) + h;
            c2 = ComputeCost(X, Y, W_try, b, lambda);

            grad_W{j}(i) = (c2-c) / h;
        end
    end
end

function [grad_b, grad_W] = ComputeGradsNumSlow(X, Y, W, b, lambda, h)

    grad_W = cell(numel(W), 1);
    grad_b = cell(numel(b), 1);

    for j=1:length(b)
        grad_b{j} = zeros(size(b{j}));

        for i=1:length(b{j})

            b_try = b;
            b_try{j}(i) = b_try{j}(i) - h;
            c1 = ComputeCost(X, Y, W, b_try, lambda);

            b_try = b;
            b_try{j}(i) = b_try{j}(i) + h;
            c2 = ComputeCost(X, Y, W, b_try, lambda);

            grad_b{j}(i) = (c2-c1) / (2*h);
        end
    end

    for j=1:length(W)
        grad_W{j} = zeros(size(W{j}));

        for i=1:numel(W{j})

            W_try = W;
            W_try{j}(i) = W_try{j}(i) - h;
            c1 = ComputeCost(X, Y, W_try, b, lambda);
            
            W_try = W;
            W_try{j}(i) = W_try{j}(i) + h;
            c2 = ComputeCost(X, Y, W_try, b, lambda);

            grad_W{j}(i) = (c2-c1) / (2*h);
        end
    end
end
