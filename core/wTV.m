function out = wTV(params,varargin)
% Linear QSM and Total Variation regularization 
% with spatially variable fidelity and regularization weights.
% This uses ADMM to solve the functional.
%
% Parameters: params - structure with 
% Required fields:
% params.input: local field map
% params.alpha1: gradient penalty (L1-norm) or regularization weight
% Optional fields:
% params.K: dipole kernel in the frequency space
% params.mu1: gradient consistency weight (ADMM weight, recommended = 100*alpha1)
% params.mu2: fidelity consistency weight (ADMM weight, recommended value = 1.0)
% params.maxOuterIter: maximum number of iterations (recommended = 150)
% params.tolUpdate: convergence limit, update ratio of the solution (recommended = 0.1)
% params.weight: data fidelity spatially variable weight (recommended = magnitude_data). 
% params.regweight: regularization spatially variable weight.
% params.isPrecond: preconditionate solution by smart initialization (default = true)
% params.isGPU: GPU acceleration (default = true)
%
% Output: out - structure with the following fields:
% out.x: calculated susceptibility map
% out.iter: number of iterations needed
% out.time: elapsed time (excluding pre-calculations)
% out.totalTime: total elapsed time (including pre-calculations)
%
% Based on the code by Bilgic Berkin at http://martinos.org/~berkin/software.html
% Modified by Carlos Milovic in 2017.03.30
% Modified by Carlos Milovic in 2020.07.07
% Last modified by Carlos Milovic and Patrich Fuchs in 2021.10.11


global DEBUG; if isempty(DEBUG); DEBUG = false; end
totalTime = tic;

[phase, alpha, N, Kernel, mu, mu2, maxOuterIter, ...
 tolUpdate, regweight, W, isGPU,isPrecond] = parse_inputs(params, varargin{:});



% Redefinition of variable for computational efficiency
Wy = (W.*phase./(W+mu2));    

% Variable initialization
z_dx = zeros(N, 'single');
z_dy = zeros(N, 'single');
z_dz = zeros(N, 'single');

s_dx = zeros(N, 'single');
s_dy = zeros(N, 'single');
s_dz = zeros(N, 'single');

x = zeros(N, 'single');

if isPrecond
    z2 = Wy; % start with something similar to the input phase, weighted to reduce noise
else
    z2 = zeros(N,'single');
end
s2 = zeros(N,'single');

alpha_over_mu = alpha/mu; % for efficiency


% Define the operators
[k1, k2, k3] = ndgrid(0:N(1)-1,0:N(2)-1,0:N(3)-1);

E1 = 1 - exp(2i .* pi .* k1 / N(1));
E2 = 1 - exp(2i .* pi .* k2 / N(2));
E3 = 1 - exp(2i .* pi .* k3 / N(3));

% Move variables to GPU
try
if isGPU 
    disp('GPU enabled');
    Wy = gpuArray(single(Wy));
    z_dx = gpuArray(z_dx);
    z_dy = gpuArray(z_dy);
    z_dz = gpuArray(z_dz);

    s_dx = gpuArray(s_dx);
    s_dy = gpuArray(s_dy);
    s_dz = gpuArray(s_dz);

    x = gpuArray(x);
    Kernel = gpuArray(Kernel);

    z2 = gpuArray(single(z2));
    s2 = gpuArray(s2);

    tolUpdate = gpuArray(tolUpdate);

    E1 = gpuArray(single(E1));
    E2 = gpuArray(single(E2));
    E3 = gpuArray(single(E3));

    alpha_over_mu = gpuArray(alpha_over_mu);
    regweight = gpuArray(single(regweight));
    mu2 = gpuArray(mu2);
    W = gpuArray(single(W));
    
end
catch
    disp('WARNING: GPU disabled');
end
    
E1t = conj(E1);
E2t = conj(E2);
E3t = conj(E3);

EE2 = E1t .* E1 + E2t .* E2 + E3t .* E3;

fprintf('%3s\t%10s\n', 'Iter', 'Update');
tic
for t = 1:maxOuterIter
    
    % update x : susceptibility estimate
    tx = E1t .* fftn(z_dx - s_dx);
    ty = E2t .* fftn(z_dy - s_dy);
    tz = E3t .* fftn(z_dz - s_dz);
    
    x_prev = x;
    x = real(ifftn( (mu * (tx + ty + tz) + mu2*conj(Kernel) .* fftn(z2-s2)) ./ (eps + mu2*abs(Kernel).^2 + mu * EE2) ));
    clear tx ty tz

    x_update = 100 * norm(x(:)-x_prev(:)) / norm(x(:));
    fprintf('%3d\t%10.4f\n', t, x_update);
    
    if x_update < tolUpdate
        break
    end
    
    if t < maxOuterIter
        % update z : gradient domain variable
        Fx = fftn(x);
        x_dx = real(ifftn(E1 .* Fx));
        x_dy = real(ifftn(E2 .* Fx));
        x_dz = real(ifftn(E3 .* Fx));
        
        z_dx = max(abs(x_dx + s_dx) - regweight(:,:,:,1)*alpha_over_mu, 0) .* sign(x_dx + s_dx);
        z_dy = max(abs(x_dy + s_dy) - regweight(:,:,:,2)*alpha_over_mu, 0) .* sign(x_dy + s_dy);
        z_dz = max(abs(x_dz + s_dz) - regweight(:,:,:,3)*alpha_over_mu, 0) .* sign(x_dz + s_dz);
    
        % update s : Lagrange multiplier
        s_dx = s_dx + x_dx - z_dx;
        s_dy = s_dy + x_dy - z_dy;            
        s_dz = s_dz + x_dz - z_dz;  
        
        clear x_dx x_dy x_dz
        % update z2 and s2 : data consistency
        z2 = Wy + mu2*real(ifftn(Kernel.*Fx)+s2)./(W + mu2);
        
        s2 = s2 + real(ifftn(Kernel.*Fx)) - z2;
        clear Fx
    end
end
% Extract output values
out.time = toc;toc
out.totalTime = toc(totalTime);
if isGPU
    out.x = gather(x);
else
    out.x = x;
end
out.iter = t;


end



function [phase, alpha, N, Kernel, mu, mu2, maxOuterIter, tolUpdate, regweight, W, isGPU, isPrecond] = parse_inputs(params, varargin)
global DEBUG; if isempty(DEBUG); DEBUG = false; end

if isstruct(params)
    if isfield(params,'input')
        phase = params.input;
    else
        error('Please provide a struct with "input" field as the local phase input.')
    end
    if isfield(params,'alpha1')
        alpha = params.alpha1;
    else
        error('Please provide a struct with "alpha1" field as input.')
    end
else
    error('Please provide a struct with "input" and "alpha1" fields as input.')
end
N = size(phase);

% Define Defaults
defaultMu = 100*alpha;
defaultMu2 = 1.0;
defaultNoOuter = 150;
defaultTol = 0.1;
defaultRegweight = ones([N 3]);
defaultW = ones(N);

p = inputParser;
p.KeepUnmatched = true;
addRequired(p,  'phase',                         @(x) isnumeric(x));
addRequired(p,  'alpha1',                         @(x) isscalar(x));
addParameter(p, 'K',            [],              @(x) isnumeric(x));
addParameter(p, 'mu1',          defaultMu,       @(x) isscalar(x));
addParameter(p, 'mu2',          defaultMu2,      @(x) isscalar(x));
addParameter(p, 'maxOuterIter', defaultNoOuter,  @(x) isscalar(x));
addParameter(p, 'tolUpdate',    defaultTol,      @(x) isscalar(x));
addParameter(p, 'regweight',    defaultRegweight,@(x) isnumeric(x));
addParameter(p, 'weight',       defaultW,        @(x) isnumeric(x));
addParameter(p, 'magnitude',    [],              @(x) isnumeric(x));
addParameter(p, 'isPrecond',    true,            @(x) islogical(x));
addParameter(p, 'isGPU',        true,            @(x) islogical(x));

if DEBUG; fprintf(1,'Parsing inputs...'); end
parse(p, phase, alpha, params, varargin{:});

phase    = single(p.Results.phase);
alpha   = single(p.Results.alpha1);

if any(strcmpi(p.UsingDefaults, 'k'))
    if isfield(params,'voxelSize')
        voxelSize = params.voxelSize;
    else
        voxelSize = [1,1,1];
    end
    if isfield(params,'B0direction')
        B0direction = params.B0direction;
    else
        B0direction = [0,0,1];
    end
    Kernel = dipole_kernel_angulated( N, voxelSize, B0direction ); 
else
    Kernel   = p.Results.K;
end

mu           = p.Results.mu1;
mu2          = p.Results.mu2;
maxOuterIter = p.Results.maxOuterIter;
tolUpdate    = p.Results.tolUpdate;

regweight = p.Results.regweight;
if not(any(strcmpi(p.UsingDefaults, 'regweight')))
    if length(size(regweight)) == 3
            regweight = repmat(regweight,[1,1,1,3]);
    end
end

magnitude = p.Results.magnitude;

if any(strcmpi(p.UsingDefaults, 'w'))
    warning('No weight supplied.')
end
if any(strcmpi(p.UsingDefaults, 'w')) && not(any(strcmpi(p.UsingDefaults, 'magnitude')))
    W = magnitude.*magnitude;
else
    W = p.Results.weight.*p.Results.weight;
end

isGPU         = p.Results.isGPU;
isPrecond     = p.Results.isPrecond;
if DEBUG; fprintf(1,'\tDone.\n'); end
end
