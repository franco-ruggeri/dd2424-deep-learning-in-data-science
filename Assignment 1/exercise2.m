clear all
close all
clc

rng(400)

global IMPROVEMENTS
IMPROVEMENTS.A = true;  % use all data
IMPROVEMENTS.B = true;  % take the best model through the epochs
IMPROVEMENTS.C = false;  % grid search
IMPROVEMENTS.D = true;  % decay learning rate
IMPROVEMENTS.E = false;  % Xavier initialization
IMPROVEMENTS.G = false;  % shuffle before each epoch

global SVM_MULTICLASS_LOSS
SVM_MULTICLASS_LOSS = true;


%% Load data

fprintf('Loading data...\n\n');

dir = 'datasets/cifar-10-batches-mat/';

if IMPROVEMENTS.A
    % training data
    TrainingSet = struct('X', [], 'Y', [], 'y', []);
    for batch = 1:5
        filename = sprintf('%sdata_batch_%d.mat', dir, batch);
        [X, Y, y] = LoadBatch(filename);
        TrainingSet.X = [TrainingSet.X X];
        TrainingSet.Y = [TrainingSet.Y Y];
        TrainingSet.y = [TrainingSet.y y];
    end

    % validation data (100 images from each class)
    ValidationSet = struct('X', [], 'Y', [], 'y', []);
    K = size(TrainingSet.Y, 1);
    for class = 1:K
        % select randomly 100 images of this class
        idx = find(TrainingSet.y == class);
        idx = idx(randperm(length(idx)));
        idx = idx(1:100);

        % add to validation set and remove from training set
        ValidationSet.X = [ValidationSet.X TrainingSet.X(:, idx)];
        ValidationSet.Y = [ValidationSet.Y TrainingSet.Y(:, idx)];
        ValidationSet.y = [ValidationSet.y TrainingSet.y(idx)];
        TrainingSet.X(:, idx) = [];
        TrainingSet.Y(:, idx) = [];
        TrainingSet.y(idx) = [];
    end
else
    [X, Y, y] = LoadBatch([dir 'data_batch_1.mat']);
    TrainingSet = struct('X', X, 'Y', Y, 'y', y);

    [X, Y, y] = LoadBatch([dir 'data_batch_2.mat']);
    ValidationSet = struct('X', X, 'Y', Y, 'y', y);
end

% test data
[X, Y, y] = LoadBatch([dir 'test_batch.mat']);
TestSet = struct('X', X, 'Y', Y, 'y', y);

% show images
I = reshape(TrainingSet.X, 32, 32, 3, size(TrainingSet.X, 2));
I = permute(I, [2, 1, 3, 4]);
% montage(I(:, :, :, 1:500), 'Size', [5,5]);


%% Pre-process data

mean_train = mean(TrainingSet.X, 2);
std_train = std(TrainingSet.X, 0, 2);

% z-score normalization
TrainingSet.X = (TrainingSet.X - mean_train) ./ std_train;
ValidationSet.X = (ValidationSet.X - mean_train) ./ std_train;
TestSet.X = (TestSet.X - mean_train) ./ std_train;


%% Initialize parameters

K = size(TrainingSet.Y, 1);
d = size(TrainingSet.X, 1);

if IMPROVEMENTS.E
    % Xavier initialization
    fan_in = d;
    fan_out = K;
    left = -sqrt(6) / (fan_in + fan_out);
    right = -left;

    W = left + (right-left) * rand(K, d);
    b = left + (right-left) * rand(K, 1);
else
    std = .01;
    W = std * randn(K, d);
    b = std * randn(K, 1);
end


%% Check gradients

% disp('Checking gradients...');
% 
% trainX = TrainingSet.X(:, 1:100);
% trainYOneHot = TrainingSet.Y(:, 1:100);
% trainY = TrainingSet.y(1:100);
% W_copy = W(:, :);
% 
% P = EvaluateClassifier(trainX, W_copy, b);
% lambda = 1;
% 
% [grad_W, grad_b] = ComputeGradients(trainX, trainYOneHot, P, W_copy, lambda);
% % [ngrad_b, ngrad_W] = ComputeGradsNum(trainX, trainYOneHot, W_copy, b, lambda, 1e-6);
% [ngrad_b, ngrad_W] = ComputeGradsNumSlow(trainX, trainYOneHot, W_copy, b, lambda, 1e-6);
% 
% fprintf('Max absolute error grad_W: %e\n', max(abs(grad_W - ngrad_W), [], 'all'));
% fprintf('Max relative error grad_W: %e\n', max(abs(grad_W - ngrad_W) ./ max(eps, abs(grad_W) + abs(ngrad_W)), [], 'all'));
% fprintf('Max absolute error grad_b: %e\n', max(abs(grad_b - ngrad_b), [], 'all'));
% fprintf('Max relative error grad_b: %e\n\n', max(abs(grad_b - ngrad_b) ./ max(eps, abs(grad_b) + abs(ngrad_b)), [], 'all'));


%% Learn

disp('Training...');

dir = 'result_pics/';

n_updates = 20000;
n = size(TrainingSet.X, 2);

if IMPROVEMENTS.C
    % grid search
    minCost = Inf;
    config = 1;
    for n_batch = linspace(1, 100, 5)
        for eta = logspace(-3, 0, 3)
            for lambda = linspace(0, 1, 20)
                GDparams = struct('n_batch', round(n_batch), 'eta', eta, 'n_epochs', round(n_updates * n_batch / n), 'lambda', lambda);
                [Wtmp, btmp] = MiniBatchGD(TrainingSet.X, TrainingSet.Y, ValidationSet.X, ValidationSet.Y, GDparams, W, b);
                
                newCost = ComputeCost(ValidationSet.X, ValidationSet.Y, Wtmp, btmp, 0);
                if newCost < minCost
                    Wstar = Wtmp;
                    bstar = btmp;
                    minCost = newCost;
                    bestGDparams = GDparams;
                end
                
                fprintf('Grid search %d settings completed\n', config);
                config = config+1;
            end
        end
    end
    
    % result of grid search
    fprintf('Best setting: n_batch=%d, eta=%.3f, lambda=%.1f\n', bestGDparams.n_batch, bestGDparams.eta, bestGDparams.lambda);
    
    % accuracy
    acc = ComputeAccuracy(TestSet.X, TestSet.y, Wstar, bstar);
    fprintf('Test accuracy: %.2f%%\n', acc*100);

    % class templates
    f = figure();
    Montage(Wstar);
    saveas(f, [dir sprintf('class_templates_%d.jpg', config)]);
else
    % parameters
    GDparams = [
        struct('n_batch', 100, 'eta', .1, 'n_epochs', round(n_updates*100/n), 'lambda', 0)
        struct('n_batch', 100, 'eta', .001, 'n_epochs', round(n_updates*100/n), 'lambda', 0)
        struct('n_batch', 100, 'eta', .001, 'n_epochs', round(n_updates*100/n), 'lambda', .1)
        struct('n_batch', 100, 'eta', .001, 'n_epochs', round(n_updates*100/n), 'lambda', 1)
    ];

    for config = 1:length(GDparams)
        % mini-batch gradient descent
        f = figure();
        [Wstar, bstar] = MiniBatchGD(TrainingSet.X, TrainingSet.Y, ValidationSet.X, ValidationSet.Y, GDparams(config), W, b);
        saveas(f, [dir sprintf('learning_curve_%d.jpg', config)]);

        % accuracy
        acc = ComputeAccuracy(TestSet.X, TestSet.y, Wstar, bstar);
        fprintf('Test accuracy %d: %.2f%%\n', config, acc*100);

        % class templates
        f = figure();
        Montage(Wstar);
        saveas(f, [dir sprintf('class_templates_%d.jpg', config)]);
    end
end


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

function P = EvaluateClassifier(X, W, b)
    global SVM_MULTICLASS_LOSS
    
    S = W*X + b;
    if SVM_MULTICLASS_LOSS
        P = S;
    else
        P = exp(S) ./ sum(exp(S));
    end
    
%     % alternative with the deep learning toolbox
%     P = softmax(W*X + b);
end

function J = ComputeCost(X, Y, W, b, lambda)
    global SVM_MULTICLASS_LOSS
    
    P = EvaluateClassifier(X, W, b);
    
    % loss for each image
    N = size(X, 2);
    l = zeros(1, N);
    if SVM_MULTICLASS_LOSS
        for n = 1:N
            y = Y(:, n) == 1;
            l(n) = sum(max(0, P(~y, n) - P(y, n) + 1));
        end
    else
        for n = 1:N
            l(n) = -log(Y(:, n)' * P(:, n));
        end
    end
    
    % regularization
    r = sum(W.^2, 'all');
    
    % cost
    J = 1 / length(l) * sum(l) + lambda * r;
end

function acc = ComputeAccuracy(X, y, W, b)
    P = EvaluateClassifier(X, W, b);
    [~, ypred] = max(P);
    nCorrect = length(find(ypred == y));
    nTot = size(X, 2);
    acc = nCorrect / nTot;
end

function [grad_W, grad_b] = ComputeGradients(X, Y, P, W, lambda)
    global SVM_MULTICLASS_LOSS    
    
    N = size(X, 2);
    K = size(Y, 1);
    
    if SVM_MULTICLASS_LOSS
        G = zeros(K, N);
        for n = 1:N
            y = find(Y(:, n) == 1);
            for c = 1:K
                if c == y
                    continue
                elseif P(c, n) - P(y, n) + 1 > 0
                    G(c, n) = G(c, n) + 1;
                    G(y, n) = G(y, n) - 1;
                end
            end
        end
    else
        G = P - Y;
    end

    grad_W = 1 / N * G * X' + 2 * lambda * W;
    grad_b = 1 / N * sum(G, 2);
end

function [Wstar, bstar] = MiniBatchGD(trainX, trainY, valX, valY, GDparams, W, b)
    global IMPROVEMENTS

    % get parameters
    n_batch = GDparams.n_batch;
    eta = GDparams.eta;
    n_epochs = GDparams.n_epochs;
    lambda = GDparams.lambda;
    n = size(trainX, 2);
    
    % stats
    costs_train = [ComputeCost(trainX, trainY, W, b, lambda), zeros(1, n_epochs)];
    costs_val = [ComputeCost(valX, valY, W, b, lambda), zeros(1, n_epochs)];
    losses_train = [ComputeCost(trainX, trainY, W, b, 0), zeros(1, n_epochs)];
    losses_val = [ComputeCost(valX, valY, W, b, 0), zeros(1, n_epochs)];
    
    % init result
    Wtmp = W;   % current network (can be worse than a previous one when overfitting)
    btmp = b;
    Wstar = W;  % best network (can be not the last one when overfitting)
    bstar = b;
    minCost = costs_val(1);
    
    for epoch = 1:n_epochs
        if IMPROVEMENTS.G
            % shuffle
            idx = randperm(n);
            trainX = trainX(:, idx);
            trainY = trainY(:, idx);
        end
        
        for batch = 1:n/n_batch
            % select minibatch
            idx_start = (batch-1) * n_batch + 1;
            idx_end = batch * n_batch;
            idx = idx_start:idx_end;
            batchX = trainX(:, idx);
            batchY = trainY(:, idx);
            
            % forward pass
            P = EvaluateClassifier(batchX, Wtmp, btmp);

            % backward pass
            [grad_W, grad_b] = ComputeGradients(batchX, batchY, P, Wtmp, lambda);
            
            % update
            Wtmp = Wtmp - eta * grad_W;
            btmp = btmp - eta * grad_b;
        end
        
        % stats
        costs_train(epoch+1) = ComputeCost(trainX, trainY, Wtmp, btmp, lambda);
        costs_val(epoch+1) = ComputeCost(valX, valY, Wtmp, btmp, lambda);
        losses_train(epoch+1) = ComputeCost(trainX, trainY, Wtmp, btmp, 0);
        losses_val(epoch+1) = ComputeCost(valX, valY, Wtmp, btmp, 0);
        
        % update best network
        if IMPROVEMENTS.B
            if costs_val(epoch+1) < minCost
                Wstar = Wtmp;
                bstar = btmp;
                minCost = costs_val(epoch+1);
            elseif abs(costs_val(epoch+1) - minCost) > 1e-3
                % early stopping
                n_epochs = epoch;   % for the plots
                break;
                
                % Alternative: remove break, continue to update but take
                % the best. More expensive but can find better solutions!
                % The validation cost can be non-monotonic!
            end
        else
            % take always the last one
            Wstar = Wtmp;
            bstar = btmp;
        end
        
        if IMPROVEMENTS.D
            % decay learning rate
            eta = .9 * eta;
            
%             % alternative: factor of 10 every 10 epochs
%             if epoch % 10 == 0
%                 eta = .1 * eta;
%             end
        end
        
%         fprintf('Epoch %d completed\n', epoch);
    end
    
    % plot learning curve
    hold on
    plot(1:n_epochs+1, costs_train(1:n_epochs+1), 'linewidth', 2);
    plot(1:n_epochs+1, costs_val(1:n_epochs+1), 'linewidth', 2);
    plot(1:n_epochs+1, losses_train(1:n_epochs+1), '--', 'linewidth', 1.5);
    plot(1:n_epochs+1, losses_val(1:n_epochs+1), '--', 'linewidth', 1.5);
    xlabel('epoch');
    ylabel('cost / loss');
    legend('training cost', 'validation cost', 'training loss', 'validation loss');
end

function Montage(images)
    % re-arrange
    s_im = cell(1, size(images, 1));
    for iImage = 1:size(images, 1)
        image = reshape(images(iImage, :), 32, 32, 3);
        s_im{iImage} = (image - min(image(:))) / (max(image(:)) - min(image(:)));
        s_im{iImage} = permute(s_im{iImage}, [2, 1, 3]);
    end

    montage(s_im, 'Size', [2, 5]);
end


%% Provided functions for numerical computation of gradients

function [grad_b, grad_W] = ComputeGradsNum(X, Y, W, b, lambda, h)

    no = size(W, 1);
    d = size(X, 1);

    grad_W = zeros(size(W));
    grad_b = zeros(no, 1);

    c = ComputeCost(X, Y, W, b, lambda);

    for i=1:length(b)
        b_try = b;
        b_try(i) = b_try(i) + h;
        c2 = ComputeCost(X, Y, W, b_try, lambda);
        grad_b(i) = (c2-c) / h;
    end

    for i=1:numel(W)   

        W_try = W;
        W_try(i) = W_try(i) + h;
        c2 = ComputeCost(X, Y, W_try, b, lambda);

        grad_W(i) = (c2-c) / h;
    end
end

function [grad_b, grad_W] = ComputeGradsNumSlow(X, Y, W, b, lambda, h)

    no = size(W, 1);
    d = size(X, 1);

    grad_W = zeros(size(W));
    grad_b = zeros(no, 1);

    for i=1:length(b)
        b_try = b;
        b_try(i) = b_try(i) - h;
        c1 = ComputeCost(X, Y, W, b_try, lambda);
        b_try = b;
        b_try(i) = b_try(i) + h;
        c2 = ComputeCost(X, Y, W, b_try, lambda);
        grad_b(i) = (c2-c1) / (2*h);
    end

    for i=1:numel(W)

        W_try = W;
        W_try(i) = W_try(i) - h;
        c1 = ComputeCost(X, Y, W_try, b, lambda);

        W_try = W;
        W_try(i) = W_try(i) + h;
        c2 = ComputeCost(X, Y, W_try, b, lambda);

        grad_W(i) = (c2-c1) / (2*h);
    end
end
