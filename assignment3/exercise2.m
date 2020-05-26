clear variables
close all
clc

rng(1);

dir_dataset = '../datasets/surnames/';
dir_result_pics = 'result_pics/';
dir_result_searches = 'result_searches/';

global DEBUG
global OPTIMIZATIONS;
global COMPENSATE_IMBALANCE;
global RANDOM_SEARCH;

DEBUG = false;
OPTIMIZATIONS.A = true;         % pre-computed M_{x,k,nf} for the first layer
OPTIMIZATIONS.B = true;         % M_{x,k} instead of M_{x,k,nf}
COMPENSATE_IMBALANCE = true;    % compensate imbalance of dataset
RANDOM_SEARCH = true;


%% Prepare data

fprintf('Preparing data...\n\n');

filename_ascii = [dir_dataset 'ascii_names.mat'];
filename_encoded = [dir_dataset 'encoded_names.mat'];
filename_validation = [dir_dataset 'Validation_Inds.txt'];
filename_languages = [dir_dataset 'category_labels.txt'];

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
    [X, Ys] = EncodeNames(all_names, ys, d, n_len, K, char_to_ind);
    save(filename_encoded, 'X', 'Ys');
else
    aux = load(filename_encoded);
    X = aux.X;
    Ys = aux.Ys;
end

% load languages (class labels)
fid = fopen(filename_languages);
languages = textscan(fid, '%*d %s');
fclose(fid);
languages = languages{1};

% partition in training and validation set
fid = fopen(filename_validation);
validation_idx = fscanf(fid, '%d');
fclose(fid);
ValidationSet.X = X(:, validation_idx);
ValidationSet.Ys = Ys(:, validation_idx);
ValidationSet.ys = ys(validation_idx)';
TrainingSet.X = X;
TrainingSet.Ys = Ys;
TrainingSet.ys = ys';
TrainingSet.X(:, validation_idx) = [];
TrainingSet.Ys(:, validation_idx) = [];
TrainingSet.ys(validation_idx) = [];

% shuffle training set
idx = 1:size(TrainingSet.X, 2);
idx = idx(randperm(length(idx)));
TrainingSet.X = TrainingSet.X(:, idx);
TrainingSet.Ys = TrainingSet.Ys(:, idx);
TrainingSet.ys = TrainingSet.ys(idx);

% examine data
y = zeros(K, 2);
for k = 1:K
    y(k, 1) = numel(find(TrainingSet.ys == k));
    y(k, 2) = numel(find(ValidationSet.ys == k));
end
x = categorical(languages);
f = figure();
bar(x, y);
set(gca,'TickLength',[0 0])
set(gca, 'YScale', 'log')
saveas(f, [dir_result_pics 'dataset.jpg']);


%% Check convolutional matrices

if DEBUG
    disp('Checking convolution matrices...');

    filename = 'DebugInfo.mat';

    debug = load(filename);

    [d_debug, k_debug, nf_debug] = size(debug.F);
    n_len_debug = size(debug.X_input, 2);

    MX = MakeMXMatrix(debug.x_input, d_debug, k_debug, nf_debug, 1);
    MF = MakeMFMatrix(debug.F, n_len_debug, 1);

    s1 = MX * debug.F(:);
    s2 = MF * debug.x_input;

    disp([s1, s2, debug.vecS]);
end


%% Check gradients

if DEBUG
    disp('Checking gradients...');

    batch_size_debug = 3;
    conv_layers_debug = [5, 5, 1, 0; 5, 5, 1, 0];
    
    TrainingSet_debug.X = TrainingSet.X(:, 1:batch_size_debug);
    TrainingSet_debug.Ys = TrainingSet.Ys(:, 1:batch_size_debug);
    TrainingSet_debug.ys = TrainingSet.ys(1:batch_size_debug);
    
    ConvNet_debug = InitConvNet(conv_layers_debug, d, n_len, K);
    if OPTIMIZATIONS.A
        ConvNet_debug.MX1 = PrecomputeMX1(TrainingSet_debug.X, ConvNet_debug.F{1}, ConvNet_debug.stride(1), ConvNet_debug.padding(1));
    end
    [P, X] = EvaluateClassifier(TrainingSet_debug.X, ConvNet_debug);

    Gs = ComputeGradients(X, TrainingSet_debug.Ys, P, ConvNet_debug);
    Gs_num = NumericalGradient(TrainingSet_debug.X, TrainingSet_debug.Ys, ConvNet_debug, 1e-6);

    for l = 1:size(conv_layers_debug, 1)
        fprintf('Max absolute error grad_F%d: %e\n', l, max(abs(Gs{l} - Gs_num{l}), [], 'all'));
        fprintf('Max relative error grad_F%d: %e\n', l, max(abs(Gs{l} - Gs_num{l}) ./ max(eps, abs(Gs{l}) + abs(Gs_num{l})), [], 'all'));
    end
    fprintf('Max absolute error grad_W: %e\n', max(abs(Gs{end} - Gs_num{end}), [], 'all'));
    fprintf('Max relative error grad_W: %e\n', max(abs(Gs{end} - Gs_num{end}) ./ max(eps, abs(Gs{end}) + abs(Gs_num{end})), [], 'all'));
    fprintf('\n');
end


%% Random search

if RANDOM_SEARCH
    disp('Random searching...');

    filename = 'random_search.txt';

    n_trials = 20;
    n_updates = 1000;
    n_smallest_class = FindSmallestClass(TrainingSet.ys, K);

    % learning rate
    eta_min = -4;
    eta_max = -4;
    eta = eta_min + (eta_max - eta_min) * rand(n_trials, 1);
    eta = 10.^eta;      % log scale

    % batch size
    batch_size_min = 100;
    batch_size_max = 100;
    batch_size = randi([batch_size_min, batch_size_max], [n_trials, 1]);

    % architecture
    n_conv_layers_min = 2;
    n_conv_layers_max = 5;
    n_conv_layers = randi([n_conv_layers_min, n_conv_layers_max], [n_trials, 1]);
    k_min = 3;
    k_max = 7;
    k = randi([k_min, k_max], [n_trials, 1]);
    nf_min = 10;
    nf_max = 100;
    nf = randi([nf_min, nf_max], [n_trials, 1]);

    % train and save results
    fileID = fopen([dir_result_searches filename], 'a');
    n = size(TrainingSet.X, 2);
    for iTrial = 1:n_trials
        % prepare parameters
        fprintf('eta=%f, batch_size=%d, k=%d, nf=%d, n_conv_layers=%d - training... ', eta(iTrial), batch_size(iTrial), k(iTrial), nf(iTrial), n_conv_layers(iTrial));
        if COMPENSATE_IMBALANCE
            epochs = round(n_updates / floor(n_smallest_class * K / batch_size(iTrial)));
        else
            epochs = round(n_updates / floor(n / batch_size(iTrial)));
        end
        conv_layers = zeros(n_conv_layers(iTrial), 4);
        conv_layers(:, 1) = k(iTrial);
        conv_layers(:, 2) = nf(iTrial);
        % stride and padding to keep same spatial dimension
        conv_layers(:, 3) = 1;
        conv_layers(:, 4) = floor((k(iTrial) - 1) / 2);
        
        % train
        GDparams = struct('eta', eta(iTrial), 'rho', .9, 'batch_size', batch_size(iTrial), 'epochs',epochs);
        ConvNet = InitConvNet(conv_layers, d, n_len, K);
        ConvNet = MiniBatchGD(TrainingSet, ValidationSet, GDparams, ConvNet);

        % test and save
        acc = ComputeAccuracy(ValidationSet.X, ValidationSet.ys, ConvNet);
        fprintf(fileID, 'eta=%f, batch_size=%d, k=%d, nf=%d, n_conv_layers=%d - validation accuracy %.2f%%\n', eta(iTrial), batch_size(iTrial), k(iTrial), nf(iTrial), n_conv_layers(iTrial), acc*100);
        fprintf('done\n');
    end
    fprintf(fileID, '\n\n\n');
    fclose(fileID);
    fprintf('\n');
    return
end


%% Train best network

disp('Training best network...');

% ConvNet architecture
% 1 row per layer with format [k, nf, stride, padding]
conv_layers = [5, 100, 1, 2; 3, 100, 1, 1];
ConvNet = InitConvNet(conv_layers, d, n_len, K);

% hyper-parameters
n_updates = 40000;
n_smallest_class = FindSmallestClass(TrainingSet.ys, K);
n = size(TrainingSet.X, 2);
batch_size = 100;
if COMPENSATE_IMBALANCE
    epochs = round(n_updates / floor(n_smallest_class * K / batch_size));
else
    epochs = round(n_updates / floor(n / batch_size));
end
GDparams = struct('eta', .001, 'rho', .9, 'batch_size', batch_size, 'epochs', epochs, 'n_update', 500);

% train
[ConvNet, f_loss, f_acc] = MiniBatchGD(TrainingSet, ValidationSet, GDparams, ConvNet);
saveas(f_loss, [dir_result_pics 'loss.jpg']);
saveas(f_acc, [dir_result_pics 'accuracy.jpg']);

% accuracy (on validation set, not test set, see instructions)
[acc, acc_per_class] = ComputeAccuracy(ValidationSet.X, ValidationSet.ys, ConvNet);
fprintf('Accuracy (validation set): %.2f%%\n', acc*100);
fprintf('Accuracy per class (validation set):\n');
for k = 1:K
    fprintf('%s: %.2f%%\n', languages{k}, acc_per_class(k)*100);
end
fprintf('\n');

% confusion matrix (on validation set, not test set, see instructions)
CM = ComputeConfusionMatrix(ValidationSet.X, ValidationSet.ys, ConvNet);
f_cm = figure();
confusionchart(CM, languages);
saveas(f_cm, [dir_result_pics 'confusion_matrix.jpg']);
% fprintf('Confusion matrix (validation set)\n');
% disp(CM);

% predict my surname and those of my friends
names = {'ruggeri', 'salmeri', 'lee', 'scofield', 'fdfr', 'gonzales'};
ys = [10, 10, 2, 5, 7, 17];
[X, Ys] = EncodeNames(names, ys, d, n_len, K, char_to_ind);
P = EvaluateClassifier(X, ConvNet);
[~, ypred] = max(P);
fprintf('Predictions of some new surnames\n');
for iName = 1:length(names)
    fprintf('%s\n', names{iName});
    fprintf('prediction: %s (probability %.2f)\n', languages{ypred(iName)}, P(ypred(iName), iName));
    fprintf('true: %s (probability %.2f)\n\n', languages{ys(iName)}, P(ys(iName), iName));
end
fprintf('Probabilities:\n');
disp(P);


%% Functions

function [X, Ys] = EncodeNames(names, ys, d, n_len, K, char_to_ind)
    N = length(names);
    X = zeros(d*n_len, N);
    
    for iName = 1:N
        name = names{iName};

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
end

function ConvNet = InitConvNet(conv_layers, d, n_len, K)
    n_conv_layers = size(conv_layers, 1);
    ConvNet.stride = zeros(n_conv_layers, 1);
    ConvNet.padding = zeros(n_conv_layers, 1);
    
    % convolutional layers
    for l = 1:n_conv_layers
        k = conv_layers(l, 1);
        nf = conv_layers(l, 2);
        ConvNet.stride(l) = conv_layers(l, 3);
        ConvNet.padding(l) = conv_layers(l, 4);
        
        % He initialization
        if l == 1
            sig = 1 / sqrt(k);  % modified for the first layer (see note)
        else
            sig = sqrt(2 / (d*k));
        end
        ConvNet.F{l} = randn(d, k, nf) * sig;
        
        % keep track for FC layer
        n_len = floor((n_len + 2*ConvNet.padding(l) - k) / ConvNet.stride(l)) + 1;
        
        % keep track for next convolutional layer
        d = nf;
    end
    
    % fully connected layer
    nf = conv_layers(end, 2);
    sig = sqrt(2 / (nf*n_len));
    ConvNet.W = randn(K, nf*n_len) * sig;
end

function MF = MakeMFMatrix(F, nlen, stride)
    [d, k, nf] = size(F);
    nlen1 = floor((nlen - k) / stride) + 1;     % width of a response map
    dk = d*k;                                   % size of a vectorized filter
    MF = zeros(nlen1*nf, nlen*d);
    VF = reshape(F, [dk, nf])';
    
    for n = 1:nlen1
        row_start = 1 + (n-1) * nf;
        row_end = row_start + nf - 1;
        col_start = 1 + (n-1) * stride * d;
        col_end = col_start + dk - 1;
        
        MF(row_start:row_end, col_start:col_end) = VF;
    end
end

function MX = MakeMXMatrix(x_input, d, k, nf, stride)
    nlen = length(x_input) / d;
    nlen1 = floor((nlen - k) / stride) + 1;     % width of a response map
    dk = d*k;                                   % size of a vectorized filter

    MX = zeros(nlen1*nf, dk*nf);
    X_input = reshape(x_input, [d, nlen]);
    
    for n = 1:nlen1
        for f = 1:nf
            row_start = (n-1) * nf + f;
            row_end = row_start;
            col_start = 1 + (f-1) * dk;
            col_end = col_start + dk - 1;
            
            col_X_start = 1 + (n-1)*stride;
            col_X_end = col_X_start + k - 1;
            aux = X_input(:, col_X_start:col_X_end);
            MX(row_start:row_end, col_start:col_end) = aux(:)';
        end
    end
end

function [P_batch, X_batch] = EvaluateClassifier(X_batch, ConvNet)
    n_conv_layers = length(ConvNet.F);
    n_len = size(X_batch, 1) / size(ConvNet.F{1}, 1);
    X_batch = [X_batch; cell(n_conv_layers, 1)];
    
    % convolutional layers
    for l = 1:n_conv_layers
        X_batch{l} = ZeroPadding(X_batch{l}, size(ConvNet.F{l}, 1), ConvNet.padding(l));
        MF = MakeMFMatrix(ConvNet.F{l}, n_len+2*ConvNet.padding(l), ConvNet.stride(l));
        X_batch{l+1} = max(0, MF * X_batch{l});
        n_len = floor((n_len + 2*ConvNet.padding(l) - size(ConvNet.F{l}, 2)) / ConvNet.stride(l)) + 1;
    end
    
    % fully connected layer
    S_batch = ConvNet.W * X_batch{end};
    expS_batch = exp(S_batch);
    P_batch = expS_batch ./ sum(expS_batch);
end

function loss = ComputeLoss(X_batch, Ys_batch, ConvNet)
    P = EvaluateClassifier(X_batch, ConvNet);
    n = size(X_batch, 2);
    
    loss = 0;
    for j = 1:n
        loss = loss - log(Ys_batch(:, j)' * P(:, j));
    end
    loss = loss / n;
end

function Gs = ComputeGradients(X_batch, Ys_batch, P_batch, ConvNet)
    global OPTIMIZATIONS
    
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
        [d, k, nf] = size(ConvNet.F{l});
        
        for j = 1:n
            g = G_batch(:, j);
            x = X_batch{l}(:, j);
            
            if OPTIMIZATIONS.A && l == 1 && isfield(ConvNet, 'MX1')
                MX = ConvNet.MX1(:, :, j);          % pre-computed M_{x,k,nf} for the first layer
            elseif OPTIMIZATIONS.B
                MX = MakeMXMatrix(x, d, k, 1, ConvNet.stride(l));      % M_{x,k} instead of M_{x,k,nf}
            else
                MX = MakeMXMatrix(x, d, k, nf, ConvNet.stride(l));
            end
            
            if OPTIMIZATIONS.B
                V = MX' * reshape(g, nf, [])';
                v = V(:);
            else
                v = g' * MX;
            end

            V = reshape(v, [d, k, nf]);
            Gs{l} = Gs{l} + 1/n * V;
        end

        % back-propagate gradient
        if l > 1
            nlen = size(X_batch{l}, 1) / d;
            MF = MakeMFMatrix(ConvNet.F{l}, nlen, ConvNet.stride(l));
            
            G_batch = MF' * G_batch;
            G_batch(X_batch{l} <= 0) = 0;
            
            % Back-propagate through zero-padding, removing rows at the
            % beginning and at the end. Theoretically, following the
            % computational graph, it should be done before the previous
            % operation, but we would need X_batch not padded. Having
            % X_batch padded, we do it here, the result is the same.
            idx_start = ConvNet.padding(l)*d + 1;
            idx_end = size(G_batch, 1) - ConvNet.padding(l)*d;
            G_batch = G_batch(idx_start:idx_end, :);
        end
    end
end

function [ConvNet, f_loss, f_acc] = MiniBatchGD(TrainingSet, ValidationSet, GDparams, ConvNet)
    global OPTIMIZATIONS;
    global COMPENSATE_IMBALANCE;
    global RANDOM_SEARCH;
    
    % hyper-parameters
    batch_size = GDparams.batch_size;
    epochs = GDparams.epochs;
    eta = GDparams.eta;                     % learning rate
    rho = GDparams.rho;                 	% momentum term
    
    % other sizes
    n = size(TrainingSet.X, 2);
    K = size(ConvNet.W, 1);
    if COMPENSATE_IMBALANCE
        n_smallest_class = FindSmallestClass(TrainingSet.ys, K);
    else
        idx_epoch = 1:n;
    end
    if COMPENSATE_IMBALANCE
        n_batches = floor(n_smallest_class * K / batch_size);
    else
        n_batches = floor(n / batch_size);
    end
    n_conv_layers = length(ConvNet.F);
    max_update = epochs * n_batches;
    
    if OPTIMIZATIONS.A
        % pre-computed MX1 (whole training set)
        MX1 = PrecomputeMX1(TrainingSet.X, ConvNet.F{1}, ConvNet.stride(1), ConvNet.padding(1));
    end
    
    if ~RANDOM_SEARCH
        % stats
        n_update = GDparams.n_update;       % compute stats every n_update
        n_measures = floor(max_update / n_update);
        if mod(max_update, n_update) ~= 0
            n_measures = n_measures + 1;
        end
        train_loss = [ComputeLoss(TrainingSet.X, TrainingSet.Ys, ConvNet), zeros(1, n_measures)];
        val_loss = [ComputeLoss(ValidationSet.X, ValidationSet.Ys, ConvNet), zeros(1, n_measures)];
        train_acc = [ComputeAccuracy(TrainingSet.X, TrainingSet.ys, ConvNet), zeros(1, n_measures)];
        val_acc = [ComputeAccuracy(ValidationSet.X, ValidationSet.ys, ConvNet), zeros(1, n_measures)];
        measured_updates = [0, zeros(1, n_measures)];
        fprintf('Confusion matrix (validation set, update %d of %d)\n', 0, max_update);
        disp(ComputeConfusionMatrix(ValidationSet.X, ValidationSet.ys, ConvNet));
        idx_measure = 2;
        count_update = 1;
    end
    
	% init momentum
    V = cell(n_conv_layers+1, 1);
    for l = 1:n_conv_layers
        V{l} = zeros(size(ConvNet.F{l}));
    end
    V{end} = zeros(size(ConvNet.W));
    
    % keep record of best network
    ConvNet_best = ConvNet;
    best_val_acc = -Inf;

    for epoch = 1:epochs
        if COMPENSATE_IMBALANCE
            % indeces for balanced training set
            idx_epoch = BalanceDataset(TrainingSet, n_smallest_class);
        end
        
        % shuffle indeces
        idx_epoch = idx_epoch(randperm(length(idx_epoch)));
        
        for batch = 1:n_batches
            % select batch
            idx_batch = SelectBatch(batch_size, batch);     % absolute batch indices
            idx_batch = idx_epoch(idx_batch);               % batch indeces relative to current training set
            X_batch = TrainingSet.X(:, idx_batch);
            Ys_batch = TrainingSet.Ys(:, idx_batch);
            if OPTIMIZATIONS.A
                ConvNet.MX1 = MX1(:, :, idx_batch);
            end
        
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

            % update best network
            aux = ComputeAccuracy(ValidationSet.X, ValidationSet.ys, ConvNet);
            if aux > best_val_acc
                ConvNet_best = ConvNet;
                best_val_acc = aux;
            end
            
            if ~RANDOM_SEARCH
                % stats
                if mod(count_update, n_update) == 0 || count_update == max_update
                    train_loss(idx_measure) = ComputeLoss(TrainingSet.X, TrainingSet.Ys, ConvNet);
                    val_loss(idx_measure) = ComputeLoss(ValidationSet.X, ValidationSet.Ys, ConvNet);
                    train_acc(idx_measure) = ComputeAccuracy(TrainingSet.X, TrainingSet.ys, ConvNet);
                    val_acc(idx_measure) = aux;
                    measured_updates(idx_measure) = count_update;
                    idx_measure = idx_measure + 1;

                    % confusion matrix
                    fprintf('Confusion matrix (validation set, iteration %d of %d)\n', count_update, max_update);
                    disp(ComputeConfusionMatrix(ValidationSet.X, ValidationSet.ys, ConvNet));
                end
                count_update = count_update + 1;
            end
        end
    end
    ConvNet = ConvNet_best;
    
    if ~RANDOM_SEARCH
        % plot loss curve
        f_loss = figure();
        hold on
        plot(measured_updates, train_loss, 'linewidth', 2);
        plot(measured_updates, val_loss, 'linewidth', 2);
        xlabel('update step');
        ylabel('loss');
        legend('training', 'validation');

        % plot accuracy
        f_acc = figure();
        hold on
        plot(measured_updates, train_acc, 'linewidth', 2);
        plot(measured_updates, val_acc, 'linewidth', 2);
        xlabel('update step');
        ylabel('accuracy');
        legend('training', 'validation');
    end
end

function [acc, acc_per_class] = ComputeAccuracy(X_batch, ys_batch, ConvNet)
    P_batch = EvaluateClassifier(X_batch, ConvNet);
    [~, ypred] = max(P_batch);
    nCorrect = length(find(ypred == ys_batch));
    nTot = size(X_batch, 2);
    acc = nCorrect / nTot;
    
    if nargout > 1
        K = size(ConvNet.W, 1);
        acc_per_class = zeros(K, 1);
        for k = 1:K
            % data belonging to class k
            idx = find(ys_batch == k);
            nTotal = length(idx);
            
            % # data belonging to class k and predicted as class k
            nCorrect = length(find(ypred(idx) == k));
            acc_per_class(k) = nCorrect / nTotal;
        end
    end
end

function CM = ComputeConfusionMatrix(X_batch, ys_batch, ConvNet)
    K = size(ConvNet.W, 1);
    P = EvaluateClassifier(X_batch, ConvNet);
    [~, ypred] = max(P);
    
    CM = zeros(K);
    for k1 = 1:K
        % data belonging to class k1
        idx = ys_batch == k1;

        for k2 = 1:K
            % # data belonging to class k1 and predicted as class k2
            n_pred = length(find(ypred(idx) == k2));
            CM(k1, k2) = n_pred;
        end
    end
end

function MX1 = PrecomputeMX1(X, F1, stride, padding)
    global OPTIMIZATIONS
    
    [d, k, nf] = size(F1);
    nlen = size(X, 1) / d;
    nlen1 = floor((nlen + 2*padding - k) / stride) + 1;     % width of a response map
    n = size(X, 2);
    if OPTIMIZATIONS.B
        nf = 1;     % M_{x,k} instead of M_{x,k,nf}
    end

    % zero-padding
    X = ZeroPadding(X, d, padding);
    
    % these matrices contain 0s and 1s, we can store booleans to occupy less memory
    MX1 = false(nlen1*nf, d*k*nf, n);
    
    for j = 1:n
        MX = MakeMXMatrix(X(:, j), d, k, nf, stride);
        MX1(:, :, j) = logical(MX);
    end
end

function batch_indeces = SelectBatch(batch_size, batch)
    idx_start = (batch-1) * batch_size + 1;
    idx_end = batch * batch_size;
    batch_indeces = idx_start:idx_end;
end

function [n_samples, class] = FindSmallestClass(ys, K)
    n_samples = Inf;
    for k = 1:K
        aux = length(find(ys == k));
        if aux < n_samples
            n_samples = aux;
            class = k;
        end
    end
end

function balanced_indeces = BalanceDataset(Dataset, n_per_class)
    K = size(Dataset.Ys, 1);
    d = size(Dataset.X, 1);
    n = K * n_per_class;
    balanced_indeces = zeros(n, 1);
    
    for k = 1:K
        % sample
        idx = find(Dataset.ys == k);
        idx = idx(randperm(length(idx), n_per_class));
        
        % add to balanced indeces
        idx_start = 1+(k-1)*n_per_class;
        idx_end = idx_start + n_per_class - 1;
        balanced_indeces(idx_start:idx_end) = idx;
    end
end

function X_batch = ZeroPadding(X_batch, d, padding)
    padding = zeros(d*padding, size(X_batch, 2));
    X_batch = [padding; X_batch; padding];
end


%% Provided function for numerical computation of gradients

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