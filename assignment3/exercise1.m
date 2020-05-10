clear all
close all
clc

rng(1);

dir_dataset = '../datasets/surnames/';


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
    
    for iName = 1:length(all_names)
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

    % save encoded names
    save(filename_encoded);
else
    % load encoded names
    aux = load(filename_encoded);
    X = aux.X;
end

% partition in training and validation set
fid = fopen(filename_validation);
validation_idx = split(fgets(fid));
fclose(fid);
validation_idx = validation_idx(1:end-1);
for iIdx = 1:length(validation_idx)
    validation_idx{iIdx} = str2double(validation_idx{iIdx});
end
validation_idx = cell2mat(validation_idx);
ValidationSet.X = X(:, validation_idx);
ValidationSet.y = ys(validation_idx);
TrainingSet.X = X;
TrainingSet.y = ys;
TrainingSet.X(:, validation_idx) = [];
TrainingSet.y(validation_idx) = [];


%% ConvNet architecture

% hyper-parameters
n1 = 3;
k1 = 2;
n2 = 1;
k2 = 2;
eta = .001;
rho = .9;

% intermediate dimensions
n_len1 = n_len - k1 + 1;
n_len2 = n_len1 - k2 + 1;

% He initialization
sig1 = 1 / sqrt(k1);
sig2 = sqrt(2 / (n1*k2));
sig3 = sqrt(2 / (n2*n_len2));
ConvNet.F{1} = randn(d, k1, n1) * sig1;
ConvNet.F{2} = randn(n1, k2, n2) * sig2;
ConvNet.W = randn(K, n2*n_len2) * sig3;


%% Check convolutional matrices

disp('Checking gradients...');

filename = 'DebugInfo.mat';

debug = load(filename);

[d_, k_, nf_] = size(debug.F);
n_len_ = size(debug.X_input, 2);

MX = MakeMXMatrix(debug.x_input, d_, k_, nf_);
MF = MakeMFMatrix(debug.F, n_len_);

s1 = MX * debug.F(:);
s2 = MF * debug.x_input;

disp([s1, s2, debug.vecS]);


%% Back-propagation

%% Mini-batch GD with momentum

%% Sampling of the training data

%% Functions for evaluation


%% Functions

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