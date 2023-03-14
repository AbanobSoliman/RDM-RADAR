%% ex 1 Maximim range
clc; clear all; close all;
%Operating frequency (Hz)
fc = 77.0e9;

%Transmitted power (W)
Ps = 3e-3;

%Antenna Gain (linear)
G =  10000;

%Minimum Detectable Power
Pe = 1e-10;

%RCS of a car
RCS = 100;

%Speed of light
c = 3*10^8;

%TODO: Calculate the wavelength
lmda = c / fc;

%TODO : Measure the Maximum Range a Radar can see. 
range = nthroot((Ps * G^2 * lmda^2 * RCS)/(Pe * (4*pi)^3), 4);
disp(range);

%% ex2 Obstacle Range
clc; clear all; close all;
range = 300;
delta_r = 1;
c = 3e8;

% TODO : Find the Bsweep of chirp for 1 m resolution
Bsweep = c / (2 * delta_r);

% TODO : Calculate the chirp time based on the Radar's Max Range
Ts = (5.5 * 2 * range) / c;

% TODO : define the frequency shifts 
fb = [0, 1.1e6, 13e6, 24e6];

% Display the calculated range
calculated_range = (c * fb * Ts) / (2 * Bsweep);
disp(calculated_range);

%% ex3 Doppler Velocity Calculation
clc; clear all; close all;
c = 3*10^8;         %speed of light
f = 77e9;   %frequency in Hz

% TODO: Calculate the wavelength
lambda = c / f; % in meters

% TODO: Define the doppler shifts in Hz using the information from above 
fd = [3e3, -4.5e3, 11e3, -3e3];

% TODO: Calculate the velocity of the targets  fd = 2*vr/lambda
vr = fd * lambda / 2;
De = 200;
Do = De - vr * 5;

% TODO: Display results
disp(vr);
disp(De);
%% ex4 Part 1 : 1D FFT
clc; clear all; close all
% Generate Noisy Signal

% Specify the parameters of a signal with a sampling frequency of 1 kHz 
% and a signal duration of 1.5 seconds.

Fs = 1000;            % Sampling frequency                    
T = 1/Fs;             % Sampling period       
L = 1500;             % Length of signal
t = (0:L-1)*T;        % Time vector

% Form a signal containing a 50 Hz sinusoid of amplitude 0.7 and a 120 Hz 
% sinusoid of amplitude 1.

S = 0.7*sin(2*pi*50*t) + sin(2*pi*120*t);

% Corrupt the signal with zero-mean white noise with a variance of 4
X = S + 2*randn(size(t));

% Plot the noisy signal in the time domain. It is difficult to identify
% the frequency components by looking at the signal X(t).

figure(1);
tiledlayout(1,2)

% left plot
nexttile
plot(1000*t(1:50), X(1:50))
title('Signal corrupted with Zero-Mean Random Noise')
xlabel('t (milliseconds)')
ylabel('X(t)')

% Compute the Fourier Transform of the Signal.

Y = fft(X);

% Compute the 2 sided spectrum P2. Then compute the single-sided spectrum
% P1 based on P2 and the even-valued signal length L.

P2 = abs(Y/L);
P1 = P2(1:L/2+1);

% Define the frequency domain f and plot the single-sided amplitude 
% spectrum P1. The amplitudes are not exactly at 0.7 and 1, as expected,
% because of the added noise. On average, longer signals produce better 
% frequency approximations

f = Fs*(0:(L/2))/L;

nexttile
plot(f,P1) 
title('Single-Sided Amplitude Spectrum of X(t)')
xlabel('f (Hz)')
ylabel('|P1(f)|')

saveas(gcf, 'fft_1d.png')

% Part 2 - 2D FFT
% Implement a second FFT along the second dimension to determine the 
% Doppler frequency shift.

% First we need to generate a 2D signal
% Convert 1D signal X to 2D using reshape

% while reshaping a 1D signal to 2D we need to ensure that dimensions match
% length(X) = M*N

% let
M = length(X)/50;
N = length(X)/30;

X_2d = reshape(X, [M, N]);

figure(2);
tiledlayout(1,2)

nexttile
imagesc(X_2d)

% Compute the 2-D Fourier transform of the data. Shift the zero-frequency 
% component to the center of the output, and plot the resulting 
% matrix, which is the same size as X_2d.

Y_2d = fft2(X_2d, M, N);

nexttile
imagesc(abs(fftshift(Y)))

saveas(gcf, 'fft_2d.png')

%% ex5 1D CA-CFAR
clc; clear all; close all

% Implement 1D CFAR using lagging cells on the given noise and target scenario.
% Generate Noisy Signal

% Specify the parameters of a signal with a sampling frequency of 1 kHz 
% and a signal duration of 1.5 seconds.

Fs = 1000;            % Sampling frequency                    
T = 1/Fs;             % Sampling period       
L = 1500;             % Length of signal
t = (0:L-1)*T;        % Time vector

% Form a signal containing a 50 Hz sinusoid of amplitude 0.7 and a 120 Hz 
% sinusoid of amplitude 1.

S = 0.7*sin(2*pi*50*t) + sin(2*pi*120*t);

% Corrupt the signal with zero-mean white noise with a variance of 4
X = S + 2*randn(size(t));

X_cfar = abs(X);

% Data_points
Ns = 1500;  % let it be the same as the length of the signal

%Targets location. Assigning bin 100, 200, 300, and 700 as Targets
%  with the amplitudes of 16, 18, 27, 22.
X_cfar([100 ,200, 300, 700])=[16 18 27 22];

% plot the output
figure(1);
tiledlayout(2,1)
nexttile
plot(X_cfar)

% Apply CFAR to detect the targets by filtering the noise.

% TODO: Define the number of Training Cells
T = 12;
% TODO: Define the number of Guard Cells 
G = 4;
% TODO: Define Offset (Adding room above noise threshold for the desired SNR)
offset = 5;

% Initialize vector to hold threshold values 
threshold_cfar = zeros(Ns-(G+T+1),1);

% Initialize Vector to hold final signal after thresholding
signal_cfar = zeros(Ns-(G+T+1),1);

% Slide window across the signal length
for i = 1:(Ns-(G+T+1))     

    % TODO: Determine the noise threshold by measuring it within 
    % the training cells
    noise_level = sum(X_cfar(i:i+T-1));   
    % TODO: scale the noise_level by appropriate offset value and take
    % average over T training cells
    threshold = noise_level / T;
    % Add threshold value to the threshold_cfar vector
    threshold_cfar(i) = offset * threshold;
    % TODO: Measure the signal within the CUT
    signal = X_cfar(i+G+T);
    % add signal value to the signal_cfar vector
     signal_cfar(i) = signal;
end

% plot the filtered signal
plot(signal_cfar);
legend('Signal')

% plot original sig, threshold and filtered signal within the same figure.
nexttile
plot(X_cfar);
hold on
plot(circshift(threshold_cfar,G),'r--','LineWidth',2)
hold on
plot (circshift(signal_cfar,(T+G)),'g--','LineWidth',2);
legend('Signal','CFAR Threshold','detection')

%% Project: 2D CA-CFAR on Range Doppler Maps (RDMs)

% The following steps can be used to implement 2D-CFAR in MATLAB:
% 
% Determine the number of Training cells for each dimension Tr and Td. Similarly, pick the number of guard cells Gr and Gd.
% Slide the Cell Under Test (CUT) across the complete cell matrix
% Select the grid that includes the training, guard and test cells. Grid Size = (2Tr+2Gr+1)(2Td+2Gd+1).
% The total number of cells in the guard region and cell under test. (2Gr+1)(2Gd+1).
% This gives the Training Cells : (2Tr+2Gr+1)(2Td+2Gd+1) - (2Gr+1)(2Gd+1)
% Measure and average the noise across all the training cells. This gives the threshold
% Add the offset (if in signal strength in dB) to the threshold to keep the false alarm to the minimum.
% Determine the signal level at the Cell Under Test.
% If the CUT signal level is greater than the Threshold, assign a value of 1, else equate it to zero.
% Since the cell under test are not located at the edges, due to the training cells occupying the edges, we suppress the edges to zero. Any cell value that is neither 1 nor a 0, assign it a zero.

