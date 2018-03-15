%% parameters
% range of composition for fitting
x = linspace(1e-4,0.2,100);
% range of T for fitting
T = 817:0.1:907;

% molar volume
Vm = 9.9e-6;

%% do the fit
[p, xeq, maxerr_phad, maxerr_xeq, maxerr_keq] = fit_parabolic_batch(x, T);


fprintf('max error in calphad xeq: %g\n', max(maxerr_phad(:)));
fprintf('max error in parabolic xeq: %g\n', max(maxerr_xeq(:)));
fprintf('max error in parabolic keq: %g\n', max(maxerr_keq(:)));


%% output

% to mat file
save('fit_result.mat','p','T','xeq','maxerr_phad','maxerr_xeq','maxerr_keq','-v7.3');

% to bin file
fname_para = 'para_coef.bin';
fname_comp = 'comp_phad.bin';

f1 = fopen(fname_para, 'w');
f2 = fopen(fname_comp, 'w');

fwrite(f1, p/Vm, 'double');
fwrite(f2, xeq, 'double');

fclose(f1);
fclose(f2);