% Note: most of this code is general, that is it can be re-used in another
% task different from synthesizing text. This comes to the cost of a less
% efficient representation in memory, because inputs and targets are kept
% separately. Unlike exercise 1, now AdaGrad takes sequences of characters
% and encodes on the fly, because the encoded inputs and targets are too
% big to take in memory.
%
% @author: Franco Ruggeri, fruggeri@kth.se

clear variables
close all
clc

rng(1);

dir_dataset = '../datasets/trump_tweets/';
dir_result_pics = 'result_pics/';
dir_result_synthesis = 'result_synthesis/';

global DEBUG
DEBUG = false;


%% Prepare data

fprintf('Preparing data...\n\n');

fn_tweets = {
    [dir_dataset 'condensed_2009.json'];
    [dir_dataset 'condensed_2010.json'];
    [dir_dataset 'condensed_2011.json'];
    [dir_dataset 'condensed_2012.json'];
    [dir_dataset 'condensed_2013.json'];
    [dir_dataset 'condensed_2014.json'];
    [dir_dataset 'condensed_2015.json'];
    [dir_dataset 'condensed_2016.json'];
    [dir_dataset 'condensed_2017.json'];
    [dir_dataset 'condensed_2018.json'];
};
fn_encoded_tweets = [dir_dataset 'tweets_encoded.mat'];

% load tweets (no retweets)
tweets = {};
for fn = fn_tweets'
    aux = jsondecode(fileread(fn{1}));
    idx = find(cell2mat(extractfield(aux, 'is_retweet')) == false);
    tweets = [tweets; extractfield(aux(idx), 'text')'];
end

% pre-process tweets
n_tweets = numel(tweets);
for n = 1:n_tweets
    % remove ending twitter link (if present)
    idx = strfind(tweets{n}, 'https://t.co');
    if ~isempty(idx)
        tweets{n} = tweets{n}(1:idx-2);     % -2 for space before link
    end
    idx = strfind(tweets{n}, 'http://t.co');
    if ~isempty(idx)
        tweets{n} = tweets{n}(1:idx-2);     % -2 for space before link
    end

    % replace &amp; with &
    tweets{n} = strrep(tweets{n}, '&amp;', '&');
    
    % remove CR
%     tweets{n} = strrep(tweets{n}, char(13), '');
end

% remove too short tweets
seq_length = 25;
idx = find(cellfun('length', tweets) < seq_length);
tweets(idx) = [];
n_tweets = numel(tweets);

% characters
start_of_tweet = '^';
end_of_tweet = char(0);
tweet_chars = [start_of_tweet, end_of_tweet, unique(strjoin(tweets))];
K = numel(tweet_chars);

% mapping
char_to_ind = containers.Map('KeyType', 'char', 'ValueType', 'int32');
ind_to_char = containers.Map('KeyType', 'int32', 'ValueType', 'char');
for k = 1:K
    char_to_ind(tweet_chars(k)) = k;
    ind_to_char(k) = tweet_chars(k);
end

if DEBUG
    fprintf('Mappings:\n');
    keys_char = keys(char_to_ind);
    for k = 1:K
        if keys_char{k} == newline && ind_to_char(char_to_ind(keys_char{k})) == newline
            fprintf('\\n -> %d\n', char_to_ind(keys_char{k}));
            fprintf('%d -> \\n\n', char_to_ind(keys_char{k}));
        elseif keys_char{k} == char(9) && ind_to_char(char_to_ind(keys_char{k})) == char(9)
            fprintf('\\t -> %d\n', char_to_ind(keys_char{k}));
            fprintf('%d -> \\t\n', char_to_ind(keys_char{k}));
        else
            fprintf('%c -> %d\n', keys_char{k}, char_to_ind(keys_char{k}));
            fprintf('%d -> %c\n', char_to_ind(keys_char{k}), ind_to_char(char_to_ind(keys_char{k})));
        end
    end
    fprintf('\n');
end

X_chars = tweets;
Y_chars = tweets;
for n = 1:n_tweets
    X_chars{n} = [start_of_tweet, X_chars{n}];
    Y_chars{n} = [Y_chars{n}, end_of_tweet];    % target is next character!
end

start_of_tweet = char_to_ind(start_of_tweet);
end_of_tweet = char_to_ind(end_of_tweet);


%% Check gradients

if DEBUG
    disp('Checking gradients...');

    % prepare RNN
    m = 5;
    RNN = InitRNN(K, m, K);
    h0 = zeros(m, 1);
    
    % select sequence
    seq_length_debug = 25;
    X_seq = X_chars{1}(1:seq_length_debug);
    Y_seq = Y_chars{1}(1:seq_length_debug);
    X_seq = Encode(X_seq, char_to_ind);
    Y_seq = Encode(Y_seq, char_to_ind);
    
    % compute gradients
    [P, H] = Forward(RNN, h0, X_seq, Y_seq);
    grads = Backward(RNN, h0, X_seq, Y_seq, H, P);
    num_grads = ComputeGradsNum(X_seq, Y_seq, RNN, 1e-4);
    
    % compare
    for f = fieldnames(grads)'
        fprintf('Max absolute error dL/d%s: %e\n', f{1}, max(abs(grads.(f{1}) - num_grads.(f{1})), [], 'all'));
        fprintf('Max relative error dL/d%s: %e\n', f{1}, max(abs(grads.(f{1}) - num_grads.(f{1})) ./ max(eps, abs(grads.(f{1})) + abs(num_grads.(f{1}))), [], 'all'));
    end
    fprintf('\n');
end


%% Train RNN

disp('Training...');

% hyper-parameters
m = 100;                    % dimensionality hidden state
GDparams.eta = .1;
GDparams.seq_length = seq_length;
GDparams.epochs = 10;
GDparams.char_to_ind_X = char_to_ind;
GDparams.char_to_ind_Y = char_to_ind;
GDparams.fn_synthesis = [dir_result_synthesis 'synthesis_while_training.txt'];
GDparams.freq_synthesis = 500;
GDparams.max_length_synthesis = 140;
GDparams.ind_to_char = ind_to_char;
GDparams.end_of_seq = end_of_tweet;
GDparams.start_of_seq = start_of_tweet;
GDparams.freq_show_loss = 100;

% training
RNN = InitRNN(K, m, K);
[RNN, f_loss] = AdaGrad(RNN, X_chars, Y_chars, GDparams);
saveas(f_loss, [dir_result_pics 'smooth_loss.jpg']);

% TODO: remove this
save('final_RNN.mat', 'RNN');


%% Synthesize text

disp('Synthesizing...');

h0 = zeros(m, 1);
x0 = zeros(K, 1);
x0(char_to_ind(start_of_tweet)) = 1;
n_max = 140;

Y = Synthesize(RNN, h0, x0, n_max, end_of_tweet);
Y_chars = Decode(Y, ind_to_char);
Y_chars(5) = end_of_tweet;
fprintf('%s\n', Y_chars);
fid = fopen([dir_result_synthesis 'synthesis_final.txt'], 'a');
fprintf(fid, '%s\n\n\n', Y_chars);


%% Functions

function RNN = InitRNN(d, m, C)
    sig = .01;

    RNN.b = zeros(m, 1);        % bias for hidden state
    RNN.c = zeros(C, 1);        % bias for outputs
    RNN.U = randn(m, d) * sig;  % weights input->hidden
    RNN.W = randn(m, m) * sig;  % weights hidden->hidden
    RNN.V = randn(C, m) * sig;  % weights hidden->output
end

function Y = Synthesize(RNN, h0, x0, n_max, end_of_seq)
    K = size(RNN.V, 1);
    Y = zeros(K, n_max);
    h = h0;
    x = x0;
    
    for t = 1:n_max
        % forward
        a = RNN.W*h + RNN.U*x + RNN.b;
        h = tanh(a);    
        o = RNN.V*h + RNN.c;
        p = softmax(o);
        
        % sample
        y = Sample(p);
        if y == end_of_seq  % end of sequence => stop here
            Y = Y(:,t-1);
            return;
        end
        Y(y,t) = 1;
        x = Y(:,t);
    end
end

function y = Sample(p)
    cp = cumsum(p);
    a = rand;
    ixs = find(cp-a > 0);
    y = ixs(1);
end

function X = Encode(X_chars, char_to_ind)
    n = length(X_chars);
    K = char_to_ind.Count;
    X = zeros(K, n);
    
    for t = 1:n
        c = X_chars(t);
        X(char_to_ind(c), t) = 1;
    end
end

function X_chars = Decode(X, ind_to_char)
    n = size(X, 2);
    X_chars = blanks(n);
    
    [x, ~] = find(X == 1);
    for t = 1:n
        X_chars(t) = ind_to_char(x(t));
    end
end

function [P, H, loss] = Forward(RNN, h0, X, Y)
    [K, tau] = size(Y);
    m = size(RNN.W, 1);
    P = zeros(K, tau);
    H = [h0, zeros(m, tau)];    % first column is h0, for convenience in code

    % forward
    for t = 1:tau
        a = RNN.W * H(:,t) + RNN.U * X(:,t) + RNN.b;
        H(:,t+1) = tanh(a);
        o = RNN.V * H(:,t+1) + RNN.c;
        P(:,t) = softmax(o);
    end
    H = H(:,2:end);             % remove h0
    
    % compute loss
    loss = 0;
    for t = 1:tau
        loss = loss - log(Y(:,t)' * P(:,t));
    end
end

function grads = Backward(RNN, h0, X, Y, H, P)
    [m, tau] = size(H);
    H = [h0, H];                        % first column is h0, for convenience in code
    G_o = P - Y;                        % dL/do_t (as columns)
    
    % gradients ouptut layer
    grads.V = G_o * H(:,2:end)';       	% dL/dV (needed for training)
    grads.c = sum(G_o, 2);            	% dL/dc (needed for training)
    
    % back-propagation to h_t and a_t
    G_a = zeros(m, tau);
    G_a(:,end) = G_o(:, end)' * RNN.V;                          % dL/dh_tau (as column)
    G_a(:,end) = G_a(:,end)' * diag(1 - H(:,end).^2);           % dL/da_tau (as column)
    for t = tau-1:-1:1
        G_a(:,t) = G_o(:,t)' * RNN.V + G_a(:,t+1)' * RNN.W;     % dL/dh_t (as column)
        G_a(:,t) = G_a(:,t)' * diag(1 - H(:,t+1).^2);           % dL/da_t (as column)
    end
    
    % gradients hidden layer
    grads.W = G_a * H(:,1:end-1)';    	% dL/dW (needed for training)
    grads.U = G_a * X';                	% dL/dU (needed for training)
    grads.b = sum(G_a, 2);            	% dL/db (needed for training)
end

function [RNN, f_loss] = AdaGrad(RNN, X_chars, Y_chars, GDparams)
    % hyper-parameters
    eta = GDparams.eta;
    seq_length = GDparams.seq_length;
    epochs = GDparams.epochs;
    
    % encoding
    char_to_ind_X = GDparams.char_to_ind_X;
    char_to_ind_Y = GDparams.char_to_ind_Y;
    
    % other sizes
    n_seq = numel(X_chars);     % number of independent sequences
    max_iter = 0;
    for n = 1:n_seq
        max_iter = max_iter + floor(length(X_chars{n}) / seq_length) * epochs;
    end
    m = size(RNN.W, 1);
    iter = 1;
    
    % synthesis
    fn_synthesis = GDparams.fn_synthesis;
    freq_synthesis = GDparams.freq_synthesis;
    max_length_synthesis = GDparams.max_length_synthesis;
    ind_to_char = GDparams.ind_to_char;
    end_of_seq = GDparams.end_of_seq;
    start_of_seq = zeros(char_to_ind_X.Count, 1);
    start_of_seq(GDparams.start_of_seq) = 1;
    fid_synthesis = fopen(fn_synthesis, 'a');
    
    % stats
    freq_show_loss = GDparams.freq_show_loss;
    smooth_loss = zeros(1, max_iter);
    
    % init sum of squares of gradients
    for f = fieldnames(RNN)'
        G.(f{1}) = 0;
    end
    
    for epoch = 1:epochs
        for seq = 1:n_seq
            n = size(X_chars{seq}, 2);
            hprev = zeros(m, 1);    % reset
            
            for e = 1:seq_length:n-seq_length+1
                % select piece of sequence
                X_seq = X_chars{seq}(:, e:e+seq_length-1);
                Y_seq = Y_chars{seq}(:, e:e+seq_length-1);

                % encode piece of sequence
                X_seq = Encode(X_seq, char_to_ind_X);
                Y_seq = Encode(Y_seq, char_to_ind_Y);
                
                % synthesis
                if iter == 1 || mod(iter, freq_synthesis) == 0
                    Y_synth = Synthesize(RNN, hprev, start_of_seq, max_length_synthesis, end_of_seq);
                    Y_synth = Decode(Y_synth, ind_to_char);
                    fprintf(fid_synthesis, 'Before iter=%d/%d\n%s\n\n\n', iter, max_iter, Y_synth);
                end

                % forward pass
                [P, H, loss] = Forward(RNN, hprev, X_seq, Y_seq);

                % backward pass
                grads = Backward(RNN, hprev, X_seq, Y_seq, H, P);

                % update parameters (AdaGrad)
                for f = fieldnames(RNN)'
                    % gradient clipping
                    grads.(f{1}) = max(min(grads.(f{1}), 5), -5);

                    % sum of squares of gradients
                    G.(f{1}) = G.(f{1}) + grads.(f{1}).^2;

                    % use effective learning rates
                    % adapted independently for each parameter!
                    eff_eta = eta ./ sqrt(G.(f{1}) + 1e-8);
                    RNN.(f{1}) = RNN.(f{1}) - eff_eta .* grads.(f{1});
                end

                % stats
                if iter == 1
                    smooth_loss(iter) = loss;
                else
                    smooth_loss(iter) = .999 * smooth_loss(iter-1) + .001 * loss;
                end
                if mod(iter, freq_show_loss) == 0
                    fprintf('iter=%d/%d, smooth_loss=%f\n', iter, max_iter, smooth_loss(iter));
                end
                
                hprev = H(:, end);
                iter = iter + 1;
            end
        end
    end
    
    % close file for synthesis
    fclose(fid_synthesis);
    
    % plot loss curve
    f_loss = figure();
    plot(smooth_loss);
    xlabel('update step');
    ylabel('smooth loss');
end

function loss = ComputeLoss(RNN, h0, X, Y)
    [~, ~, loss] = Forward(RNN, h0, X, Y);
end


%% Provided function for numerical computation of gradients

function num_grads = ComputeGradsNum(X, Y, RNN, h)
    for f = fieldnames(RNN)'
%         disp('Computing numerical gradient for')
%         disp(['Field name: ' f{1} ]);
        num_grads.(f{1}) = ComputeGradNum(X, Y, f{1}, RNN, h);
    end
end

function grad = ComputeGradNum(X, Y, f, RNN, h)
    n = numel(RNN.(f));
    grad = zeros(size(RNN.(f)));
    hprev = zeros(size(RNN.W, 1), 1);
    for i=1:n
        RNN_try = RNN;
        RNN_try.(f)(i) = RNN.(f)(i) - h;
        l1 = ComputeLoss(RNN_try, hprev, X, Y);
        RNN_try.(f)(i) = RNN.(f)(i) + h;
        l2 = ComputeLoss(RNN_try, hprev, X, Y);
        grad(i) = (l2-l1)/(2*h);
    end
end