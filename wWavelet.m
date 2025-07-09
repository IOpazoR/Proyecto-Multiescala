function out = wWavelet(params, varargin)
% Linear QSM with Wavelet L1 regularization using ADMM
%
% Solves: 
%    min_x (1/2)|| W ∘ (Kx - φ) ||_2^2 + α ||Wx||_1
% using ADMM in the wavelet domain.
%
% Required in params:
%   params.input:         local field map (φ)
%   params.alpha1:        regularization weight (α)
% Optional:
%   params.K:             dipole kernel in Fourier space
%   params.mu1:           regularization ADMM weight (≈ 100×alpha1)
%   params.mu2:           fidelity ADMM weight (≈ 1.0)
%   params.maxOuterIter:  max number of iterations (default = 150)
%   params.tolUpdate:     relative stopping threshold (default = 0.1)
%   params.weight:        data fidelity weight (e.g. magnitude image)
%   params.regweight:     (not used)
%   params.waveletName:   e.g. 'db4'
%   params.levels:        number of decomposition levels
%   params.isPrecond:     (default = true)
%   params.isGPU:         (default = true)

global DEBUG; if isempty(DEBUG); DEBUG = false; end
totalTime = tic;

% === Parse input ===
[phase, alpha, N, Kernel, mu, mu2, maxOuterIter, tolUpdate, ...
 regweight, W, isGPU, isPrecond, waveletName, niveles] = parse_inputs(params, varargin{:});

epsilon = 1e-8;

% === Preconditioning ===
Wy = (W .* phase) ./ (W + mu2);
x = zeros(N, 'single');

if isPrecond
    z2 = Wy;
else
    z2 = zeros(N, 'single');
end
s2 = zeros(N, 'single');
alpha_over_mu = alpha / mu;

% === Wavelet Initialization ===
coeffs = wavedec3(x, niveles, waveletName);
s_wavelet = coeffs;
z_wavelet = coeffs;
for i = 1:length(coeffs.dec)
    s_wavelet.dec{i} = zeros(size(coeffs.dec{i}), 'like', coeffs.dec{i});
    z_wavelet.dec{i} = zeros(size(coeffs.dec{i}), 'like', coeffs.dec{i});
end

% === Move to GPU if requested ===
try
    if isGPU
        disp('GPU enabled');
        Wy = gpuArray(single(Wy));
        Kernel = gpuArray(Kernel);
        z2 = gpuArray(single(z2));
        s2 = gpuArray(s2);
        W = gpuArray(single(W));
        x = gpuArray(single(x));
        alpha_over_mu = gpuArray(alpha_over_mu);
        mu2 = gpuArray(mu2);
        tolUpdate = gpuArray(tolUpdate);
        for i = 1:length(z_wavelet.dec)
            z_wavelet.dec{i} = gpuArray(z_wavelet.dec{i});
            s_wavelet.dec{i} = gpuArray(s_wavelet.dec{i});
        end
    end
catch
    disp('WARNING: GPU disabled');
end

% === ADMM iterations ===
fprintf('%3s\t%10s\n', 'It', 'Update');
tic;
for t = 1:maxOuterIter
    % Step 1: x_wavelet from current wavelet variables
    wavelet_struct = z_wavelet;
    for i = 1:length(z_wavelet.dec)
        wavelet_struct.dec{i} = z_wavelet.dec{i} - s_wavelet.dec{i};
    end
    x_wavelet = waverec3(wavelet_struct);

    % Step 2: Update x
    x_prev = x;
    Fx_wavelet = fftn(x_wavelet);
    numerator = mu * Fx_wavelet + mu2 * conj(Kernel) .* fftn(z2 - s2);
    denominator = epsilon + mu2 * abs(Kernel).^2 + mu;
    x = real(ifftn(numerator ./ denominator));

    % Step 3: Check convergence
    den = norm(x(:));
    x_update = (den > 0) * 100 * norm(x(:) - x_prev(:)) / (den + eps);
    fprintf('%3d\t%10.4f\n', t, x_update);
    if x_update < tolUpdate
        break
    end

    % Step 4: Update wavelet coefficients via soft thresholding
    coeffs = wavedec3(x + x + waverec3(s_wavelet), niveles, waveletName);
    for i = 1:length(coeffs.dec)
        temp = coeffs.dec{i} + s_wavelet.dec{i};
        z_wavelet.dec{i} = max(abs(temp) - alpha_over_mu, 0) .* sign(temp);
        s_wavelet.dec{i} = s_wavelet.dec{i} + coeffs.dec{i} - z_wavelet.dec{i};
    end

    % Step 5: Update data consistency variables
    Fx = fftn(x);
    z2 = Wy + mu2 * (real(ifftn(Kernel .* Fx)) + s2) ./ (W + mu2);
    s2 = s2 + real(ifftn(Kernel .* Fx)) - z2;
end

% === Output ===
out.time = toc;
out.totalTime = toc(totalTime);
out.iter = t;
if isGPU
    out.x = gather(x);
else
    out.x = x;
end

end


function [phase, alpha, N, Kernel, mu, mu2, maxOuterIter, tolUpdate, ...
          regweight, W, isGPU, isPrecond, nombrewavelet, niveles] = parse_inputs(params, varargin)

global DEBUG; if isempty(DEBUG); DEBUG = false; end

if isstruct(params)
    if isfield(params,'input')
        phase = params.input;
    else
        error('Please provide a struct with "input" field as the local phase input.');
    end
    if isfield(params,'alpha1')
        alpha = params.alpha1;
    else
        error('Please provide a struct with "alpha1" field as input.');
    end
else
    error('Please provide a struct with "input" and "alpha1" fields as input.');
end

N = size(phase);

% Defaults
defaultMu = 100 * alpha;
defaultMu2 = 1.0;
defaultNoOuter = 150;
defaultTol = 0.1;
defaultRegweight = ones([N 3]);
defaultW = ones(N);
defaultWaveletName = 'db4';
defaultNiveles = 3;

p = inputParser;
p.KeepUnmatched = true;

% Required
addRequired(p, 'phase', @(x) isnumeric(x));
addRequired(p, 'alpha1', @(x) isscalar(x));

% Optional
addParameter(p, 'K', [], @(x) isnumeric(x));
addParameter(p, 'mu1', defaultMu, @(x) isscalar(x));
addParameter(p, 'mu2', defaultMu2, @(x) isscalar(x));
addParameter(p, 'maxOuterIter', defaultNoOuter, @(x) isscalar(x));
addParameter(p, 'tolUpdate', defaultTol, @(x) isscalar(x));
addParameter(p, 'regweight', defaultRegweight, @(x) isnumeric(x));
addParameter(p, 'weight', defaultW, @(x) isnumeric(x));
addParameter(p, 'magnitude', [], @(x) isnumeric(x));
addParameter(p, 'isPrecond', true, @(x) islogical(x));
addParameter(p, 'isGPU', true, @(x) islogical(x));
addParameter(p, 'nombrewavelet', defaultWaveletName, @(x) ischar(x) || isstring(x));
addParameter(p, 'niveles', defaultNiveles, @(x) isnumeric(x) && isscalar(x));

if DEBUG; fprintf(1, 'Parsing inputs...'); end
parse(p, phase, alpha, params, varargin{:});

% Outputs
phase = single(p.Results.phase);
alpha = single(p.Results.alpha1);

% Dipole kernel
if any(strcmpi(p.UsingDefaults, 'k'))
    if isfield(params, 'voxelSize')
        voxelSize = params.voxelSize;
    else
        voxelSize = [1,1,1];
    end
    if isfield(params, 'B0direction')
        B0direction = params.B0direction;
    else
        B0direction = [0,0,1];
    end
    Kernel = dipole_kernel_angulated(N, voxelSize, B0direction); 
else
    Kernel = p.Results.K;
end

% ADMM parameters
mu           = p.Results.mu1;
mu2          = p.Results.mu2;
maxOuterIter = p.Results.maxOuterIter;
tolUpdate    = p.Results.tolUpdate;

% Regularization weights
regweight = p.Results.regweight;
if ~any(strcmpi(p.UsingDefaults, 'regweight'))
    if ndims(regweight) == 3
        regweight = repmat(regweight, [1,1,1,3]);
    end
end

% Fidelity weights
magnitude = p.Results.magnitude;
if any(strcmpi(p.UsingDefaults, 'weight'))
    warning('No fidelity weight supplied.');
end

if any(strcmpi(p.UsingDefaults, 'weight')) && ~any(strcmpi(p.UsingDefaults, 'magnitude'))
    W = magnitude .* magnitude;
else
    W = p.Results.weight .* p.Results.weight;
end

% Flags
isGPU     = p.Results.isGPU;
isPrecond = p.Results.isPrecond;

% Wavelet settings
nombrewavelet = char(p.Results.nombrewavelet);  % for compatibility
niveles     = p.Results.niveles;

if DEBUG; fprintf(1, '\tDone.\n'); end
end
