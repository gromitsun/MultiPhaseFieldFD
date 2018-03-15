function xeq = calc_compeq(a2, a1, a0, b2, b1, b0, lb, ub)

a = a2 * (b2 - a2)/2;
b = a2 * (b1 - a1);
c = -(a1 - b1)^2./2 - b2 * (a0 - b0);


xeq1 = [(-b-sqrt(b^2-4*a*c))/(2*a), (-b+sqrt(b^2-4*a*c))/(2*a)];
xeq1 = xeq1((xeq1>=lb(1)) & (xeq1<=ub(1)));
xeq2 = ((a1 - b1) + a2*xeq1)/b2;


xeq = [xeq1; xeq2];


% fa = @(x,T)((0.5*a2*x+a1)*x+a0);
% f1a = @(x,T)(a2*x+a1);
% fb = @(x,T)((0.5*b2*x+b1)*x+b0);
% f1b = @(x,T)(b2*x+b1);
% 
% xeq = calc_phad(fa, fb, f1a, f1b, 800, lb, ub);