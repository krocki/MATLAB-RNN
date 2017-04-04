%
% rnn.m
%
% Simple RNN implementation
% based on
% http://karpathy.github.io/2015/05/21/rnn-effectiveness/
% https://gist.github.com/karpathy/d4dee566867f8291f086
%
% Author: Kamil Rocki <kmrocki@us.ibm.com>
% Created on: 02/02/2016
%

%TODO: dropout?

%read raw byte stream
data = read_raw('enwik3.txt');

text_length = size(data, 1);

% alphabet
symbols = unique(data);
alphabet_size = size(symbols, 1);
ASCII_SIZE = 256;

% n_in - size of the alphabet, ex. 4 (ACGT)
n_in = ASCII_SIZE;
% n_out = n_in - predictions have the same size as inputs
n_out = n_in;

codes = eye(n_in);

max_iterations = text_length;
max_epochs = 1000;

observations = zeros(1, text_length);
perc = round(text_length / 100);
show_every = 5; %show stats every show_every s

% hyperparameters
hidden_size = 10; % size of hidden layer of neurons
seq_length = 3; % number of steps to unroll the RNN for
learning_rate = 1e-1;
vocab_size = n_in;

% model parameters
Wxh = randn(hidden_size, vocab_size) * 0.01; % input to hidden
Whh = randn(hidden_size, hidden_size) * 0.01; % hidden to hidden
Why = randn(vocab_size, hidden_size) * 0.01; % hidden to output
bh = zeros(hidden_size, 1); % hidden bias
by = zeros(vocab_size, 1); % output bias

% for adagrad
mWxh = zeros(size(Wxh));
mWhh = zeros(size(Whh));
mWhy = zeros(size(Why));
mbh = zeros(size(bh));
mby = zeros(size(by));

h = zeros(hidden_size, seq_length);
target = zeros(vocab_size, seq_length);
y = zeros(vocab_size, seq_length);
dy = zeros(vocab_size, seq_length);
probs = zeros(vocab_size, seq_length);

%using log2 (bits), initial guess
smooth_loss = - log2(1.0 / alphabet_size);
loss_history = [];

%reset timer
tic

for e = 1:max_epochs
 
    
    %set some random context
    h(:, 1) = clip(randn(size(Why, 2), 1) * 0.5, -1, 1);
    %or zeros
    %h(:, 1) = zeros(size(h(:, 1)));

    beginning = randi([2 1+seq_length]); %randomize starting point
    for ii = beginning:seq_length:max_iterations - seq_length
     
        % reset grads
        dWxh = zeros(size(Wxh));
        dWhh = zeros(size(Whh));
        dWhy = zeros(size(Why));
        dbh = zeros(size(bh));
        dby = zeros(size(by));
        dhnext = zeros(size(h(:, 1)));
     
        % get next symbol
        xs(:, 1:seq_length) = codes(data(ii - 1:ii + seq_length - 2), :)';
        target(:, 1:seq_length) = codes(data(ii:ii + seq_length - 1), :)';
     
        observations = char(data(ii - 1:ii + seq_length - 2))';
        t_observations = char(data(ii:ii + seq_length - 1))';
     
        % forward pass:
     
        loss = 0;
     
        for t = 2:seq_length
         
            % update h
            h(:, t) = tanh(Wxh * xs(:, t) + Whh * h(:, t - 1) + bh);
         
            % update y
            y(:, t) = Why * h(:, t) + by;
         
            % compute probs
            probs(:, t) = exp(y(:, t)) ./ sum(exp(y(:, t)));
         
            % cross-entropy loss, sum logs of probabilities of target outputs
            % i.e. for target          [  0   0   0   1   0]
            %           probs           [0.1 0.2 0.1 0.3 0.3]
            %           -log2(p)      [3.3 2.3 3.3 1.7 1.7]
            %           probs.*log2(p) [  0   0   0 1.7   0]
            %           sum                = 1.7
         
            loss = loss + sum(- log2(probs(:, t)) .* target(:, t));
         
        end
     		
     	%bits/symbol
     	loss = loss/seq_length;

        % backward pass:
        for t = seq_length: - 1:2
         
            % dy (global error)
            dy(:, t) = probs(:, t) - target(:, t); % %dy[targets[t]] -= 1 # backprop into y
            dWhy = dWhy + dy(:, t) * h(:, t)'; %dWhy += np.dot(dy, hs[t].T)
            dby = dby + dy(:, t); % dby += dy
            dh = Why' * dy(:, t) + dhnext; %dh = np.dot(Why.T, dy) + dhnext
         
            % backprop through tanh nonlinearity
            dhraw = (1 - h(:, t) .* h(:, t)) .* dh; % dhraw = (1 - hs[t] * hs[t]) * dh
            dbh = dbh + dhraw; % dbh += dhraw
            dWxh = dWxh + dhraw * xs(:, t)'; %dWxh += np.dot(dhraw, xs[t].T)
            dWhh = dWhh + dhraw * h(:, t - 1)'; %dWhh += np.dot(dhraw, prev_h)
            dhnext = Whh' * dhraw; %dhnext = np.dot(Whh.T, dhraw)
         
        end
     
        rnn_grad_check;

        % for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
        % np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
     
        dWxh = clip(dWxh, - 5, 5);
        dWhh = clip(dWhh, - 5, 5);
        dWhy = clip(dWhy, - 5, 5);
        dbh = clip(dbh, - 5, 5);
        dby = clip(dby, - 5, 5);
     
        % adjust weights, adagrad:
        mWxh = mWxh + dWxh .* dWxh;
        mWhh = mWhh + dWhh .* dWhh;
        mWhy = mWhy + dWhy .* dWhy;
        mby = mby + dby .* dby;
        mbh = mbh + dbh .* dbh;
     
        Wxh = Wxh - learning_rate * dWxh ./ (sqrt(mWxh + eps));
        Whh = Whh - learning_rate * dWhh ./ (sqrt(mWhh + eps));
        Why = Why - learning_rate * dWhy ./ (sqrt(mWhy + eps));
        bh = bh - learning_rate * dbh ./ (sqrt(mbh + eps));
        by = by - learning_rate * dby ./ (sqrt(mby + eps));
     
        % %%%%%%%%%%%%%%%%%%%%%
     
        smooth_loss = smooth_loss * 0.999 + loss * 0.001;
         
        elapsed = toc;
     
        % show stats every show_every s
        if (elapsed > show_every)
         
            loss_history = [loss_history smooth_loss];
         
            fprintf('[epoch %d] %d %% text read... smooth loss = %.3f\n', e, round(100 * ii / text_length), smooth_loss);
            fprintf('\n\nGenerating some text...\n');
           
            % random h seed
            t = generate_rnn(Wxh, Whh, Why, bh, by, 1000, clip(randn(size(Why, 2), 1) * 0.5, -1, 1));
            % generate according to the last seen h
            %t = generate_rnn(Wxh, Whh, Why, bh, by, 500, h(:, seq_length));
            fprintf('%s \n', t);
         
            % update plots
            figure(1)
            subplot(3, 3, 1);
            imagesc(target);
            str = sprintf('targets: [%s]', t_observations);
            title(str);
            subplot(3, 3, 2);
            imagesc(((h + 1) / 2), [0 1]);
            title('h');
            subplot(3, 3, 3);
            imagesc(probs, [0 1]);
            title('probs');
            subplot(3, 3, 7);
            imagesc(Wxh');
            title('Wxh');
            subplot(3, 3, 8);
            imagesc(Whh);
            title('Whh');
            subplot(3, 3, 9);
            imagesc(Why);
            title('Why');
            subplot(3, 3, 4);
            plot(loss_history);
            title('Loss history');
            subplot(3, 3, 5);
            imagesc(dy);
            title('dy');
            subplot(3, 3, 6);
            imagesc(dWhy);
            title('dWhy');
         
            drawnow;
         
            % reset timer
            tic
         
        end

        %carry
        h(:, 1) = h(:, seq_length);
        y(:, 1) = y(:, seq_length);
        probs(:, 1) = probs(:, seq_length);

    end
 
end