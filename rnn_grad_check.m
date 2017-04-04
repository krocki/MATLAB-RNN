%
% rnn_grad_check.m
%
% gradient check for RNN code 
%
% Author: Kamil Rocki <kmrocki@us.ibm.com>
% Created on: 02/04/2016
%

dby_err = zeros(vocab_size, 1);
dWhy_err = zeros(vocab_size, hidden_size);
bh_err = zeros(hidden_size, 1);
dWxh_err = zeros(hidden_size, vocab_size);
dWhh_err = zeros(hidden_size, hidden_size);
increment = 1e-3;

%dby
for k=1:vocab_size
	delta = zeros(vocab_size, 1);
	delta(k) = increment;

	pre_loss = rnn_forward(xs, target, Wxh, Whh, Why, bh, by - delta, seq_length, h(:, 1));
	post_loss = rnn_forward(xs, target, Wxh, Whh, Why, bh, by + delta, seq_length, h(:, 1));

	numerical_grad = (post_loss - pre_loss) / (increment * 2);
	analitic_grad = dby(k);
	dby_err(k) = abs(numerical_grad - analitic_grad)/abs(numerical_grad + analitic_grad);

end

%dWhy
for k=1:vocab_size
	for kk=1:hidden_size

		delta = zeros(vocab_size, hidden_size);
		delta(k, kk) = increment;
		
		pre_loss = rnn_forward(xs, target, Wxh, Whh, Why - delta, bh, by, seq_length, h(:, 1));
		post_loss = rnn_forward(xs, target, Wxh, Whh, Why + delta, bh, by, seq_length, h(:, 1));

		numerical_grad = (post_loss - pre_loss) / (increment * 2);
		analitic_grad = dWhy(k, kk);
		dWhy_err(k, kk) = abs(numerical_grad - analitic_grad)/abs(numerical_grad + analitic_grad);

	end
end

%bh
for k=1:hidden_size

	delta = zeros(hidden_size, 1);
	delta(k) = increment;
	
	pre_loss = rnn_forward(xs, target, Wxh, Whh, Why, bh - delta, by, seq_length, h(:, 1));
	post_loss = rnn_forward(xs, target, Wxh, Whh, Why, bh + delta, by, seq_length, h(:, 1));

	numerical_grad = (post_loss - pre_loss) / (increment * 2);
	analitic_grad = dbh(k);
	bh_err(k) = abs(numerical_grad - analitic_grad)/abs(numerical_grad + analitic_grad);

end

%dWxh
for kk=1:vocab_size
	for k=1:hidden_size

		delta = zeros(hidden_size, vocab_size);
		delta(k, kk) = increment;
		
		pre_loss = rnn_forward(xs, target, Wxh - delta, Whh, Why, bh, by, seq_length, h(:, 1));
		post_loss = rnn_forward(xs, target, Wxh + delta, Whh, Why, bh, by, seq_length, h(:, 1));

		numerical_grad = (post_loss - pre_loss) / (increment * 2);
		analitic_grad = dWxh(k, kk);
		dWxh_err(k, kk) = abs(numerical_grad - analitic_grad)/abs(numerical_grad + analitic_grad);

	end
end

%dWhh
for k=1:hidden_size
	for kk=1:hidden_size

		delta = zeros(hidden_size, hidden_size);
		delta(k, kk) = increment;
		
		pre_loss = rnn_forward(xs, target, Wxh, Whh - delta, Why, bh, by, seq_length, h(:, 1));
		post_loss = rnn_forward(xs, target, Wxh, Whh + delta, Why, bh, by, seq_length, h(:, 1));

		numerical_grad = (post_loss - pre_loss) / (increment * 2);
		analitic_grad = dWhh(k, kk);
		dWhh_err(k, kk) = abs(numerical_grad - analitic_grad)/abs(numerical_grad + analitic_grad);

	end
end

fprintf('dby err = %.9f\n', max(dby_err(:)));
fprintf('dWhy err = %.9f\n', max(dWhy_err(:)));
fprintf('dbh err = %.9f\n', max(bh_err(:)));
fprintf('dWxh err = %.9f\n', max(dWxh_err(:)));
fprintf('dWhh err = %.9f\n', max(dWhh_err(:)));

