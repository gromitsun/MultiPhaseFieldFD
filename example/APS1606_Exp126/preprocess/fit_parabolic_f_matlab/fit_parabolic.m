function [p, xeq, maxerr_phad, maxerr_xeq, maxerr_keq] = fit_parabolic(x, T, x0, verbose, do_plot)

ya_exact = fa(x,T);
yb_exact = fb(x,T);

if nargin < 4
    verbose = true;
    do_plot = true;
end

if (nargin < 3) || (isempty(x0))
    x0 = [polyfit(x,ya_exact,2) polyfit(x,yb_exact,2)];
    x0([1 4]) = x0([1 4])*2;
end

% lower and upper bounds for solving common tangent
lb1 = [min(x(:)) min(x(:))];
ub1 = [max(x(:)) max(x(:))];

% calculate comp_eq from exact free energy functions
if nargout > 2
    [xeq, maxerr_phad] = calc_phad(@fa,@fb,@f1a,@f1b,T,lb1,ub1,verbose);
else
    xeq = calc_phad(@fa,@fb,@f1a,@f1b,T,lb1,ub1,verbose);
end
yeq = [fa(xeq(1),T); fb(xeq(2),T)];
keq = (yeq(1)-yeq(2))/(xeq(1)-xeq(2));

% build cost function
fun = @(p) (sum((ya_exact-poly_eval(p(1:3),x)).^2) ...
    + sum(yb_exact-poly_eval(p(4:6),x)).^2);

cost_start = fun(x0(:));

% do the optimization
% A = [-1 0 0 0 0 0; 0 0 0 -1 0 0];
% b = [0; 0];

A = [];
b = [];
lb = [0 -inf -inf 0 -inf -inf];
ub = [];
Aeq = [xeq(1) 1 0 0 0 0; 0 0 0 xeq(2) 1 0; 0.5*xeq(1)^2 xeq(1) 1 0 0 0; 0 0 0 0.5*xeq(2)^2 xeq(2) 1];
beq = [keq; keq; yeq(1); yeq(2)];

% [p, fval] = fmincon(fun, x0(:), A, b, Aeq, beq, lb, ub, @(p)fminconstr(p,xeq,yeq));
[p, fval] = fmincon(fun, x0(:), A, b, Aeq, beq, lb, ub);

if (nargout > 3) || verbose
    xeq_parabolic = calc_compeq(p(1),p(2),p(3),p(4),p(5),p(6),0,0.2);
    ka = p(1)*xeq(1)+p(2);
    kb = p(4)*xeq(2)+p(5);
end

if nargout > 3
    maxerr_xeq = max(abs(xeq - xeq_parabolic));
    maxerr_keq = max(abs([ka-kb, ka-keq, kb-keq]));
end

if verbose
    fprintf('Cost at start = %g, cost at end = %g\n', cost_start, fval);
    fprintf('Xeq exact = %.15g, %.15g, Xeq parabolic = %.15g, %.15g\n', xeq, xeq_parabolic);
    fprintf('Keq = %.15g, ka = %.15g, kb = %.15g\n', keq, ka, kb);
end

if do_plot
    % x = linspace(1e-4,1-1e-4,100);
    % ya_exact = fa(x,T);
    % yb_exact = fb(x,T);
    plot(x, ya_exact, 'b--');
    hold on
    plot(x, yb_exact, 'r--');
    plot(x, poly_eval(p(1:3), x), 'b-');
    plot(x, poly_eval(p(4:6), x), 'r-');
    plot(xeq(1), yeq(1), 'bo', 'MarkerFaceColor', 'b');
    plot(xeq(2), yeq(2), 'ro', 'MarkerFaceColor', 'r');
    hold off
end

end

%% polynomial evaluation
function out = poly_eval(p, x)

out = (0.5*p(1)*x+p(2)).*x + p(3);

end

% %% nonlinear constraint function
% function [c, ceq] = fminconstr(p, xeq, yeq)

% c = [];
% ceq = [poly_eval(p(1:3),xeq(1))-yeq(1); poly_eval(p(4:6),xeq(2))-yeq(2)];

% end

