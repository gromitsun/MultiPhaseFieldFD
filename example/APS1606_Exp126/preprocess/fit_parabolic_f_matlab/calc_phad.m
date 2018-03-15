function [xeq, maxerr] = calc_phad(fa, fb, f1a, f1b, T, lb, ub, verbose)
% Calculate equilibrium compositions by common tangent construction. 
% Return maximum error in tangent slopes.

if nargin < 8
    verbose = true;
end

fun = @(x)((f1a(x(1),T)-f1b(x(2),T))^2+((fa(x(1),T)-fb(x(2),T))/(x(1)-x(2))- f1a(x(1),T))^2);
x0 = [0.01, 0.2];
A = [1, -1];
b = 0;

xeq = fmincon(fun, x0(:), A, b, [], [], lb, ub, @(x)fminconstr(x,T));
% xeq = fmincon(fun, x0(:), A, b, [], [], lb, ub);

if (nargout == 2) || verbose
    ka = f1a(xeq(1),T);
    kb = f1b(xeq(2),T);
    kab = (fa(xeq(1),T)-fb(xeq(2),T))/(xeq(1)-xeq(2));
end
    
if (nargout == 2)
    maxerr = max(abs([ka-kb, ka-kab, kb-kab]));
end

if verbose
    fprintf('f1a = %.15g, f1b = %.15g, (fa - fb)/(xa-xb) = %.15g\n', ka, kb, kab);
    fprintf('xeq1 = %.15g, xeq2 = %.15g\n', xeq);
end

end


function [c, ceq] = fminconstr(x,T)

c = [];
ceq = [f1a(x(1),T)-f1b(x(2),T);  (fa(x(1),T)-fb(x(2),T))/(x(1)-x(2)) - f1a(x(1),T)];

end