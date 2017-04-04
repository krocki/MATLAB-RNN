function text = generate_rnn(Wxh, Whh, Why, bh, by, l, h)

%
% generate_rnn.m
%
% Generates text given weights Wxh, Whh, Why, bh and by
% l - length
% h - hidden state used as a seed
%
% Simple RNN implementation
% based on
% http://karpathy.github.io/2015/05/21/rnn-effectiveness/
% https://gist.github.com/karpathy/d4dee566867f8291f086
%
% Author: Kamil Rocki <kmrocki@us.ibm.com>
% Created on: 02/02/2016
%

	text = [];
	codes = eye(size(Why, 1));

	for i=1:l-1

		y = Why * h + by;
		probs = exp(y)./sum(exp(y));
		cdf = cumsum(probs);

		r = rand();
		sample = min(find(r <= cdf));
		text = [text char(sample)];

		%update hidden state
		x = codes(sample, :)';
		h = tanh(Wxh * x + Whh * h + bh);

	end

end