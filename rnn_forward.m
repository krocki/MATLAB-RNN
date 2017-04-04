%
% rnn_forward.m
%
% rnn forward pass for grad check
%
% Author: Kamil Rocki <kmrocki@us.ibm.com>
% Created on: 02/04/2016
%

function loss = rnn_forward(xs, target, Wxh, Whh, Why, bh, by, seq_length, h_prev)

    hidden_size = size(Wxh, 1);
    vocab_size = size(Wxh, 2);

    h = zeros(hidden_size, seq_length);
    y = zeros(vocab_size, seq_length);
    probs = zeros(vocab_size, seq_length);

    h(:, 1) = h_prev;

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
     
        loss = loss + sum(- log(probs(:, t)) .* target(:, t));
     
    end