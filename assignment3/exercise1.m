clear all
close all
clc

dir_dataset = '../datasets/surnames/';


%% Prepare data

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
n1 = ;
k1 = ;
n2 = ;
k2 = ;
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


%% Back-propagation

%% Mini-batch GD with momentum

%% Sampling of the training data

%% Functions for evaluation


