function [p, xeq, maxerr_phad, maxerr_xeq, maxerr_keq] = fit_parabolic_batch(x, T, verbose)

if nargin < 3
    verbose = false;
end

x0 = [];
do_plot = false;

nT = numel(T);
p = zeros(6, nT);
xeq = zeros(2, nT);
maxerr_phad = zeros(1, nT);
maxerr_xeq = zeros(1, nT);
maxerr_keq = zeros(1, nT);

for i = 1:nT
    fprintf('Fitting T = %g ...\n', T(i));
    [p(:,i), xeq(:,i), maxerr_phad(i), maxerr_keq(i), maxerr_xeq(i)] = fit_parabolic(x, T(i), x0, verbose, do_plot);
end
    