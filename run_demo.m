function out = run_demo()

load spatial_res;           % voxel size
load msk;                   % brain mask => obtained by eroding the BET mask by 5 voxels (by setting peel=5 in LBV)


load magn;                  % magnitude from transversal orientation
load chi_cosmos;            % COSMOS from 12 orientations (in ppm)

N = size(chi_cosmos);

center = N/2 + 1;

TE = 25e-3;
B0 = 2.8936;
gyro = 2*pi*42.58;

phs_scale = TE * gyro * B0;

% Create dipole kernel and susceptibility to field model
kernel = dipole_kernel_fansi( N, spatial_res, 0 ); % 0 for the continuous kernel by Salomir and Marques and Bowtell.


% Create raw phase
rphase = ifftn(fftn(chi_cosmos).*kernel);

signal = magn.*exp(1i*phs_scale*(rphase))+0.01*(randn(size(magn))+1i*randn(size(magn))); %SNR = 40
phase_use = angle(signal)/phs_scale;
mask_use = msk;
magn_use = magn.*mask_use;

clear magn msk signal rphase

params = [];

params.K = kernel;
params.input = mask_use.*phase_use*phs_scale;
params.weight = single(mask_use);

alpha = 8e-15;
params.alpha1 = alpha;
params.mu1 = 10*alpha;
out = wWavelet(params);

end

