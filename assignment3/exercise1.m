clear all
close all
clc

rng(1);

dir_dataset = '../datasets/surnames/';
dir_result_pics = 'result_pics/';


%% Prepare data

fprintf('Preparing data...\n\n');

filename_ascii = [dir_dataset 'ascii_names.mat'];
filename_encoded = [dir_dataset 'encoded_names.mat'];
filename_validation = [dir_dataset 'Validation_Inds.txt'];

% load ascii names
if ~isfile(filename_ascii)
    ExtractNames
else
    aux = load(filename_ascii);
    all_names = aux.all_names;
    ys = aux.ys;
end

% get dimensions
C = unique(cell2mat(all_names));
d = numel(C);
n_len = max(cellfun('length', all_names));
K = numel(unique(ys));
N = numel(all_names);

% create mapping
char_to_ind = containers.Map(num2cell(C), 1:length(C));

% encode
if ~isfile(filename_encoded)
    X = zeros(d*n_len, N);
    
    % encode names
    for iName = 1:N
        name = all_names{iName};

        % one-hot encoding
        encoded_name = zeros(d, n_len);
        for iChar = 1:length(name)
            char = name(iChar);
            ind = char_to_ind(char);
            encoded_name(ind, iChar) = 1;
        end

        % vectorize
        vectorized_name = encoded_name(:);

        % store
        X(:, iName) = vectorized_name;
    end
    
    % encode labels
    Ys = zeros(K, N);
    for n = 1:N
        Ys(ys(n), n) = 1;
    end

    % save
    save(filename_encoded, 'X', 'Ys');
else
    aux = load(filename_encoded);
    X = aux.X;
    Ys = aux.Ys;
end

% partition in training and validation set
fid = fopen(filename_validation);
validation_idx = split(fgets(fid));
fclose(fid);
validation_idx = validation_idx(1:end-1);   % the last one is a space
for iIdx = 1:length(validation_idx)
    validation_idx{iIdx} = str2double(validation_idx{iIdx});
end
validation_idx = cell2mat(validation_idx);
ValidationSet.X = X(:, validation_idx);
ValidationSet.ys = ys(validation_idx);
ValidationSet.Ys = Ys(:, validation_idx);
TrainingSet.X = X;
TrainingSet.ys = ys;
TrainingSet.Ys = Ys;
TrainingSet.X(:, validation_idx) = [];
TrainingSet.ys(validation_idx) = [];
TrainingSet.Ys(:, validation_idx) = [];


%% ConvNet architecture

% 1 row per layer with format (k, nf) = (width, number of filter)
conv_layer_sizes = [
    5 10
    5 10
];


%% Check convolutional matrices

disp('Checking convolution matrices...');

filename = 'DebugInfo.mat';

debug = load(filename);

[d_, k_, nf_] = size(debug.F);
n_len_ = size(debug.X_input, 2);

MX = MakeMXMatrix(debug.x_input, d_, k_, nf_);
MF = MakeMFMatrix(debug.F, n_len_);

s1 = MX * debug.F(:);
s2 = MF * debug.x_input;

disp([s1, s2, debug.vecS]);


%% Check gradients

disp('Checking gradients...');

n_batch_ = 3;

trainX = TrainingSet.X(:, 1:n_batch_);
trainYs = TrainingSet.Ys(:, 1:n_batch_);

ConvNet = InitConvNet(conv_layer_sizes, d, n_len, K);
[P, X] = EvaluateClassifier(trainX, ConvNet);

Gs = ComputeGradients(X, trainYs, P, ConvNet);
Gs_num = NumericalGradient(trainX, trainYs, ConvNet, 1e-6);

for l = 1:size(conv_layer_sizes, 1)
    fprintf('Max absolute error grad_F%d: %e\n', l, max(abs(Gs{l} - Gs_num{l}), [], 'all'));
    fprintf('Max relative error grad_F%d: %e\n', l, max(abs(Gs{l} - Gs_num{l}) ./ max(eps, abs(Gs{l}) + abs(Gs_num{l})), [], 'all'));
end
fprintf('Max absolute error grad_W: %e\n', max(abs(Gs{end} - Gs_num{end}), [], 'all'));
fprintf('Max relative error grad_W: %e\n', max(abs(Gs{end} - Gs_num{end}) ./ max(eps, abs(Gs{end}) + abs(Gs_num{end})), [], 'all'));
fprintf('\n');


%% Train

GDparams = struct('eta', .001, 'rho', .9, 'n_batch', 100, 'max_iter', 1000, 'n_update', 50);
ConvNet = InitConvNet(conv_layer_sizes, d, n_len, K);
[ConvNet, f_loss] = MiniBatchGD(TrainingSet, ValidationSet, GDparams, ConvNet);
saveas(f_loss, [dir_result_pics 'loss.jpg']);


%% Init parameters

function ConvNet = InitConvNet(conv_layer_sizes, d, n_len, K)
    % convolutional layers
    for l = 1:size(conv_layer_sizes)
        k = conv_layer_sizes(l, 1);
        nf = conv_layer_sizes(l, 2);
        
        % He initialization
        if l == 1
            sig = 1 / sqrt(k);  % modified for the first layer (see note)
        else
            sig = sqrt(2 / (n_len*k));
        end
        ConvNet.F{l} = randn(d, k, nf) * sig;
        
        n_len = n_len - k + 1;  % keep track for FC layer
        d = nf;                 % keep track for next convolutional layer
    end
    
    % fully connected layer
    nf = conv_layer_sizes(end, 2);
    sig = sqrt(2 / (nf*n_len));
    ConvNet.W = randn(K, nf*n_len) * sig;
end


%% Convolutional matrices

function MF = MakeMFMatrix(F, nlen)
    [d, k, nf] = size(F);
    nlen1 = nlen - k + 1;   % width of a response map
    dk = d*k;              % size of a vectorized filter
    MF = zeros(nlen1*nf, nlen*d);
    VF = reshape(F, [dk, nf])';
    
    for n = 1:nlen1
        row_start = 1 + (n-1) * nf;
        row_end = row_start + nf - 1;
        col_start = 1 + (n-1) * d;
        col_end = col_start + dk - 1;
        
        MF(row_start:row_end, col_start:col_end) = VF;
    end
end

function MX = MakeMXMatrix(x_input, d, k, nf)
    nlen = length(x_input) / d;
    nlen1 = nlen - k + 1;   % width of a response map
    dk = d*k;               % size of a vectorized filter

    MX = zeros((nlen-k+1)*nf, dk*nf);
    X_input = reshape(x_input, [d, nlen]);
    
    for n = 1:nlen1
        for f = 1:nf
            row_start = (n-1) * nf + f;
            row_end = row_start;
            col_start = 1 + (f-1) * dk;
            col_end = col_start + dk - 1;
            
            aux = X_input(:, n:(n+k-1));
            MX(row_start:row_end, col_start:col_end) = aux(:)';
        end
    end
end


%% Mini-batch GD with momentum

function [P_batch, X_batch] = EvaluateClassifier(X_batch, ConvNet)
    n_conv_layers = length(ConvNet.F);
    n_len = size(X_batch, 1) / size(ConvNet.F{1}, 1);
    X_batch = [X_batch; cell(n_conv_layers-1, 1)];
    
    % convolutional layers
    for l = 1:n_conv_layers
        MF = MakeMFMatrix(ConvNet.F{l}, n_len);   
        X_batch{l+1} = max(0, MF * X_batch{l});
        n_len = n_len - size(ConvNet.F{l}, 2) + 1;
    end

    % fully connected layer
    S_batch = ConvNet.W * X_batch{end};
    expS_batch = exp(S_batch);
    P_batch = expS_batch ./ sum(expS_batch);
end

function loss = ComputeLoss(X_batch, Ys_batch, ConvNet)
    P = EvaluateClassifier(X_batch, ConvNet);
    
    % loss for each sample
    N = size(X_batch, 2);
    l = zeros(1, N);
    for n = 1:N
        l(n) = -log(Ys_batch(:, n)' * P(:, n));
    end
    
    % loss for batch
    loss = 1 / length(l) * sum(l);
end

function Gs = ComputeGradients(X_batch, Ys_batch, P_batch, ConvNet)
    n = size(X_batch{1}, 2);
    n_conv_layers = length(ConvNet.F);
    Gs = cell(n_conv_layers+1, 1);
    G_batch = P_batch - Ys_batch;

    % fully connected layer
    Gs{end} = 1 / n * G_batch * X_batch{end}';
    
    % back-propagate gradient
    G_batch = ConvNet.W' * G_batch;
    G_batch(X_batch{end} <= 0) = 0;     % same as multiplying by Ind(H(l)>0)
    
    % convolutional layers
    for l = n_conv_layers:-1:1
        Gs{l} = zeros(size(ConvNet.F{l}));
        for j = 1:n
            g = G_batch(:, j);
            x = X_batch{l}(:, j);
            
            if l == 1 && isfield(ConvNet, 'MX')
                MX = ConvNet.MX1(:, :, j);      % optimization 1: use pre-computed matrix
                v = g' * MX;
            else  
                [d, k, nf] = size(ConvNet.F{l});
                MX = MakeMXMatrix(x, d, k, 1);  % optimization 2: M_{x,k} instead of M_{x,k,nf}
                V = MX' * reshape(g, nf, [])';
                v = V(:);
            end

            V = reshape(v, [d, k, nf]);
            Gs{l} = Gs{l} + 1 / n * V;
        end

        % back-propagate gradient
        if l > 1
            nlen = size(X_batch{l}, 1) / size(ConvNet.F{l}, 1);
            MF = MakeMFMatrix(ConvNet.F{l}, nlen);
            
            G_batch = MF' * G_batch;
            G_batch(X_batch{l} <= 0) = 0;
        end
    end
end

function [ConvNet, f_loss] = MiniBatchGD(TrainingSet, ValidationSet, GDparams, ConvNet)
    n = size(TrainingSet.X, 2);
    n_conv_layers = length(ConvNet.F);

    % optimization 1: pre-compute MX for the first layer
%     [d, k, nf] = size(ConvNet.F{1});
%     nlen = size(TrainingSet.X, 1) / d;
%     ConvNet.MX1 = zeros((nlen-k+1)*nf, d*k*nf, n);
%     for j = 1:n
%         ConvNet.MX1(:, :, j) = MakeMXMatrix(TrainingSet.X(:, j), d, k1, n1);
%     end
    
    % get hyper-parameters
    n_batch = GDparams.n_batch;     % size of mini-batches
    n_batches = floor(n/n_batch);   % number of mini-batches in the training set
    eta = GDparams.eta;
    rho = GDparams.rho;
    max_iter = GDparams.max_iter;
    n_update = GDparams.n_update;   % compute stats every n_update

    % stats
    n_measures = floor(max_iter / n_update);
    losses_train = [ComputeLoss(TrainingSet.X, TrainingSet.Ys, ConvNet), zeros(1, n_measures)];
    losses_val = [ComputeLoss(ValidationSet.X, ValidationSet.Ys, ConvNet), zeros(1, n_measures)];
    measured_updates = [0, zeros(1, floor(max_iter/n_update))];
    idx_measure = 2;
    
    % init momentum
    V = cell(n_conv_layers+1, 1);
    for l = 1:n_conv_layers
        V{l} = zeros(size(ConvNet.F{l}));
    end
    V{end} = zeros(size(ConvNet.W));
    
    for t = 1:max_iter
        batch = mod(t-1, n_batches) + 1;

        % select minibatch
        idx_start = (batch-1) * n_batch + 1;
        idx_end = batch * n_batch;
        idx = idx_start:idx_end;
        X_batch = TrainingSet.X(:, idx);
        Ys_batch = TrainingSet.Ys(:, idx);

        % forward pass
        [P_batch, X_batch] = EvaluateClassifier(X_batch, ConvNet);

        % backward pass
        Gs = ComputeGradients(X_batch, Ys_batch, P_batch, ConvNet);

        % update convolutional layers
        for l = 1:n_conv_layers
            V{l} = rho * V{l} + eta * Gs{l};
            ConvNet.F{l} = ConvNet.F{l} - V{l};
        end
        
        % update fully connected layer
        V{end} = rho * V{end} + eta * Gs{end};
        ConvNet.W = ConvNet.W - V{end};

        % stats
        if mod(t, n_update) == 0
            % loss
            losses_train(idx_measure) = ComputeLoss(TrainingSet.X, TrainingSet.Ys, ConvNet);
            losses_val(idx_measure) = ComputeLoss(ValidationSet.X, ValidationSet.Ys, ConvNet);
            measured_updates(idx_measure) = t;
            idx_measure = idx_measure + 1;
            
            % confusion matrix
            fprintf('Confusion matrix (iteration %d of %d)\n', t, max_iter);
            disp(ComputeConfusionMatrix(ValidationSet.X, ValidationSet.ys, ConvNet));
        end
        
%         fprintf('Iteration %d of %d\n completed', t, max_iter);
    end
    
    % plot cost/loss curve
    f_loss = figure();
    hold on
    plot(measured_updates, losses_train, 'linewidth', 2);
    plot(measured_updates, losses_val, 'linewidth', 2);
    xlabel('update step');
    ylabel('loss');
    legend('training', 'validation');
end


%% Evaluation metrics

function acc = ComputeAccuracy(X, y, ConvNet)
    P = EvaluateClassifier(X, ConvNet);
    [~, ypred] = max(P);
    nCorrect = length(find(ypred == y));
    nTot = size(X, 2);
    acc = nCorrect / nTot;
end

function CM = ComputeConfusionMatrix(X, y, ConvNet)
    K = size(ConvNet.W, 1);
    
    P = EvaluateClassifier(X, ConvNet);
    [~, ypred] = max(P);
    
    CM = zeros(K);
    for k1 = 1:K
        % data belonging to class k1
        idx = find(y == k1);
        
        for k2 = 1:K
            % # data belonging to class k1 and predicted as class k2
            n_pred = length(find(ypred(idx) == k2));
            CM(k1, k2) = n_pred;
        end
    end
end


%% Numerical computation of gradients (provided)

function Gs = NumericalGradient(X_inputs, Ys, ConvNet, h)
    try_ConvNet = ConvNet;
    Gs = cell(length(ConvNet.F)+1, 1);

    for l=1:length(ConvNet.F)
        try_convNet.F{l} = ConvNet.F{l};

        Gs{l} = zeros(size(ConvNet.F{l}));
        nf = size(ConvNet.F{l},  3);

        for i = 1:nf        
            try_ConvNet.F{l} = ConvNet.F{l};
            F_try = squeeze(ConvNet.F{l}(:, :, i));
            G = zeros(numel(F_try), 1);

            for j=1:numel(F_try)
                F_try1 = F_try;
                F_try1(j) = F_try(j) - h;
                try_ConvNet.F{l}(:, :, i) = F_try1; 

                l1 = ComputeLoss(X_inputs, Ys, try_ConvNet);

                F_try2 = F_try;
                F_try2(j) = F_try(j) + h;            

                try_ConvNet.F{l}(:, :, i) = F_try2;
                l2 = ComputeLoss(X_inputs, Ys, try_ConvNet);            

                G(j) = (l2 - l1) / (2*h);
                try_ConvNet.F{l}(:, :, i) = F_try;
            end
            Gs{l}(:, :, i) = reshape(G, size(F_try));
        end
    end

    % compute the gradient for the fully connected layer
    W_try = ConvNet.W;
    G = zeros(numel(W_try), 1);
    for j=1:numel(W_try)
        W_try1 = W_try;
        W_try1(j) = W_try(j) - h;
        try_ConvNet.W = W_try1; 

        l1 = ComputeLoss(X_inputs, Ys, try_ConvNet);

        W_try2 = W_try;
        W_try2(j) = W_try(j) + h;            

        try_ConvNet.W = W_try2;
        l2 = ComputeLoss(X_inputs, Ys, try_ConvNet);            

        G(j) = (l2 - l1) / (2*h);
        try_ConvNet.W = W_try;
    end
    Gs{end} = reshape(G, size(W_try));
end