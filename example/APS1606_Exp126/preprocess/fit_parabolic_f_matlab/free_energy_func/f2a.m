function ret = f2a(xcu, T)
R=8.3145;
a2 = -68100. + 4.*T;
a3 = 86540. - 4.*T;
a4 = -4680.0;
ret = R*T/(xcu.*(1-xcu)) + 2*a2 + 6*a3 * xcu + 12*a4 * xcu.^2;