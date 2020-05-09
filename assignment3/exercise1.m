clear all
close all
clc

dir_dataset = '../datasets/surnames/';


%% Prepare data

filename_ascii = [dir_dataset 'ascii_names.mat'];
filename_encoded = [dir_dataset 'encoded_names.mat'];

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


%% Back-propagation

%% Mini-batch GD with momentum

%% Sampling of the training data

%% Functions for evaluation


