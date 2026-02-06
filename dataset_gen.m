clear;
clc;

%%% function declarations %%%
% generates a raw binary data
function data = generate_data(M, numSymbols)
    % compute parameters
    bitsPerSym = log2(M);
    numBits = bitsPerSym * numSymbols;

    % generate a random binary sequence
    data = randi([0 1], numBits, 1);
end

% converts binary data into M-QAM signal
function txCarrier = sample_data(data, fs, fc, M, symbolRate)
    % convert binary signal into M-QAM signal
    qamSymbols = qammod(data, M, 'InputType', 'bit', 'UnitAveragePower', true);
    
    % upsample and pulse shape
    samplesPerSymbol = round(fs / symbolRate);
    txSignal = upfirdn(qamSymbols, rcosdesign(0.35, 4, samplesPerSymbol), samplesPerSymbol);
    
    % create the time vector
    t = (0:length(txSignal)-1)' / fs;
    
    % modulate up to carrier frequency (1 GHz)
    txCarrier = real(txSignal .* exp(1j*2*pi*fc*t));
end

% split and store the signal into certain sizes
function dir = split_store_sample(idx, data, len)
    % create a new directory if the directory doesn't exist
    dir = sprintf('%0.3d', idx)
    if(exist(dir, 'dir'))
        dir = '000';
        return;
    end
    mkdir(dir);

    % split the sample and store
    for i = 1:len:length(data)
        % compute filename
        filename = sprintf('/%0.3d.csv', uint16(i/len)+1);
        filename = append(dir, filename);

        % store the splitted data
        writematrix(data(i:min([i+len-1, length(data)])), filename);
    end
end

% generate a point anomaly with the size of 2T
function gaus_kernel = generate_gaus(T, shape, k)
    % when T=0, inject impulse noise with randomized amplitude
    if T == 0
        gaus_kernel = 0.75 + k * (rand - 0.5);
        return;
    end

    % noise parameters
    sigma = T/3;
    x = -T:T;

    % bell shape template
    org_gaus = exp(-x.^2 / (2*sigma^2));

    % select shape of the noise
    switch(shape)
        case 0
            gaus_kernel = org_gaus;
        case 1
            gaus_kernel = (-x / sigma^2) .* org_gaus;
        case 2
            gaus_kernel = (x.^2/sigma^4 - 1/sigma^2) .* org_gaus;
    end

    % normalize the kernel
    gaus_kernel = gaus_kernel ./ norm(gaus_kernel);

    % add randomization
    err = k * (rand(size(gaus_kernel)) - 0.5);
    gaus_kernel = gaus_kernel + err;
end

% select the duration of the noise
function duration = select_shape()
    % generate a random number
    num = rand;

    % select the shape by range
    if(num < 1/32)
        % (T=11): 3.125 (%)
        duration = 5;
    elseif(num < 1/16)
        % (T=9): 3.125 (%)
        duration = 4;
    elseif(num < 1/8)
        % (T=7): 6.25 (%)
        duration = 3;
    elseif(num < 1/4)
        % (T=5): 12.5 (%)
        duration = 2;
    elseif(num < 1/2)
        % (T=3): 25 (%)
        duration = 1;
    else
        % (T=1): 50 (%)
        duration = 0;
    end
end

% inject point anomalies to a single csv file
function [corrupted_data, count] = add_point_to_data(data, data_len)
    % set the maximum amount of point anomalies to 1% of the data length
    N = randi([1, ceil(length(data)/100)]);

    % utilize a mask for avoid overlapping of noise
    mask = zeros(size(data));

    % utilize standard deviation of the signal for noise amplitude
    % noise_amp = std(data) * (0.5 + 0.1 * (rand - 0.5));
    % noise_amp = 10 * std(data);
    % noise_amp = (max(data) - min(data));

    % randomly select N indices
    rnd_index = randperm(numel(data), N);

    % count variable for # of noises
    count = 0;

    % copy the clean data to the result
    corrupted_data = data;

    % iteratively inject noise
    for idx = rnd_index
        % select the duration and shape of the noise
        duration = select_shape();
        shape = randi([0 2]);

        % set the range of noise
        idx_min = max([1 idx-duration]);
        idx_max = min([idx+duration length(data)]);
        noise_idx_min = (duration+1)-(idx-idx_min);
        noise_idx_max = (duration+1)+(idx_max-idx);

        % check if the range is already occupied
        is_occupied = ismember(1, mask(idx_min:idx_max));
        if(is_occupied)
            % skip noise injection for the index
            continue;
        end

        % generate noise
        k = 0.1;        % random noise coefficient to the point anomaly kernel
        gaus_kernel = generate_gaus(duration, shape, k);

        % inject noise to signal
        a = data(idx_min:idx_max);
        b = gaus_kernel(noise_idx_min:noise_idx_max)';

        % determine noise coefficients and the polarity
        noise_coef = 0.2 + (0.1 * rand);
        polar = randi([0 1]);
        if(polar == 0)
            noise_coef = -noise_coef;
        end
        corrupted_data(idx_min:idx_max) = a + noise_coef * b;

        % update the mask
        mask(idx_min:idx_max) = 1;

        % update noise count
        count = count + 1;
    end
end

% inject point anomalies to all the csv files referred by the directory
function count = add_point_anomalies(idx, data_len)
    % check if the directory exists
    dir_name = sprintf('%0.3d', idx);
    if(~exist(dir_name, 'dir'))
        % directory doesn't exist
        disp('directory does not exist');
        return;
    end

    % create a new directory
    new_dir = sprintf('%0.3d/', idx+1);
    mkdir(new_dir);

    % iterate through all the csv file under the directory
    files = dir(fullfile(dir_name, '*.csv'));
    for file_index = 1:length(files)
        % load the csv file
        file_name = append('/',files(file_index).name);
        file_name = append(dir_name, file_name);
        data = readmatrix(file_name);

        % inject point anomaly
        [corrupted_data, count] = add_point_to_data(data, data_len);

        % DEBUG
        %fprintf('Injected anomaly count(%s): %d\n', file_name, count);

        % setup a new filename
        new_file_name = append(new_dir, files(file_index).name);

        % store the data to csv file
        writematrix(corrupted_data, new_file_name);
    end
end

% invert the samples in time domain for data augmentation
function flip_samples(src_idx)
    % check source directory
    src_dir = sprintf('%0.3d', src_idx);
    if(~exist(src_dir, 'dir'))
        return;
    end

    % create new target directory
    new_dir = sprintf('%0.3d/', src_idx+2);
    mkdir(new_dir);

    % iterate through all the csv file under the source directory
    files = dir(fullfile(src_dir, '*.csv'));
    for file_index = 1:length(files)
        % load the csv file
        file_name = append('/',files(file_index).name);
        file_name = append(src_dir, file_name);
        data = readmatrix(file_name);

        % flip the data
        flipped_data = flip(data);

        % setup a new filename
        new_file_name = append(new_dir, files(file_index).name);

        % store the data to csv file
        writematrix(flipped_data, new_file_name);
    end
end

% zero padding the splitted data (leading / trailing)
function zero_padding(src_idx, data_len)
    % check source directory
    src_dir = sprintf('%0.3d', src_idx);
    if(~exist(src_dir, 'dir'))
        return;
    end

    % iterate through all the csv files under the source directory
    files = dir(fullfile(src_dir, '*.csv'));
    for file_index = 1:length(files)
        % load the csv file
        file_name = append('/', files(file_index).name);
        file_name = append(src_dir, file_name);
        data = readmatrix(file_name);

        % skip if the length(data) == data_len
        if(length(data) == data_len)
            continue;
        end

        % divide trailing and leading zero cases
        new_data = zeros(data_len, 1);
        if(mod(src_idx-1, 4) < 2)
            % create zero-trailing vector
            new_data(1:length(data)) = data(:);
        else
            % create zero-leading vector
            new_data(data_len-length(data)+1:data_len) = data(:);
        end

        % store teh data to csv file
        writematrix(new_data, file_name);        
    end
end

% % inject point anomalies to the input signal (M-QAM signal)
% function result = add_point_anomalies(data, N)
%     % copy the original data
%     result = data;

%     % get the standard deviation of the signal (used for noise amplitude
%     % calculation)
%     data_std = std(result);

%     % get N amount of indices to inject anomalies
%     rnd_index = randperm(numel(result), N);

%     % iteratively add noise
%     for idx = rnd_index
%         % determine the amplitude of the peak
%         k = 0.5 * rand + 0.25;

%         % randomize the polarity
%         polar = randi([0 1]);
%         if(polar == 0)
%             result(idx) = result(idx) + k * data_std;
%         else
%             result(idx) = result(idx) - k * data_std;
%         end
%     end
% end

function disp_sample(filename)
    % read csv file
    data = readmatrix(filename);

    % plot data
    figure;
    plot(data(1:length(data)));
    xlabel('Time (ns)');
    ylabel('Amplitude');
    title('1 GHz 16-QAM Signal');
    grid on;
end

% parameters for signal generation
M = 16;             % order of QAM
numSymbols = 1000;  % number of symbols in the data
fs = 5e9;           % sampling frequency (i.e. 5 GHz)
fc = 1e9;           % carrier frequency (i.e. 1 GHz)
symbolRate = 100e6; % symbol rate (i.e. 100 MHz)

% parameters for dataset generation
total_signals = 25;
data_len = 1000;

% iteratively generate data
for src_idx = 1:4:4*(total_signals)
    % generate random binary signal
    data = generate_data(M, numSymbols);

    % sample the signal with certain sampling frequency
    sample = sample_data(data, fs, fc, M, symbolRate);

    % split and store the data (splits original template)
    dir = split_store_sample(src_idx, sample, data_len);

    % add sequence anomalies and stores to the (4N-3) file
    count = add_point_anomalies(src_idx, data_len);

    % flip the original and noisy samples
    flip_samples(src_idx);
    flip_samples(src_idx+1);

    % zero pad all the related data
    zero_padding(src_idx, data_len);
    zero_padding(src_idx+1, data_len);
    zero_padding(src_idx+2, data_len);
    zero_padding(src_idx+3, data_len);
end

% DEBUG
disp_sample('001/050.csv');
fprintf('Amount of injected point anomaly: %d\n', count);
fprintf('max-min of data: %d\n', max(data)-min(data));