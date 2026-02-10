clear;
clc;

% activation function %
function v_out = activation_func(v_in, v_b, v_th, W, L)
    % define output size
    v_out = zeros(size(v_in));

    % define output values
    for i = 1:length(v_out)
        % voltage difference
        v_diff = v_in(i) - (v_b + abs(v_th));

        % activation interval
        if v_diff > 0
            v_out(i) = (v_diff ^ (4/5)) * (((v_b + abs(v_th)) - (15*pi/16)*(W/L)*(sqrt(2*v_in(i))))^(1/5));
        end
    end
end

% define domain and parameters %
v_in = -5:0.01:5;
v_b = 0.5;
v_th = -0.6;
W = 2.0;
L = 1.0;

% compute output
v_out = activation_func(v_in, v_b, v_th, W, L);

figure;
plot(v_in, v_out);
xlabel('Input voltage (V)');
ylabel('Output voltage (V)');
title('Hardware Activation Function');
grid on;