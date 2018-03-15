function ret = fb(xcu, T)
R=8.3145;

a0 = -271.195 + 74092./T + 0.018532*T.^2 - 5.76423e-6*T.^3 + 7.9337e-20*T.^7 + (211.207 - 38.5844*log(T))*T;
a1 = -31740.5 - 21614./T - 0.0211888*T.^2 + 5.89345e-6*T.^3 - 8.51859e-20*T.^7 + (-88.6363 + 14.472*log(T))*T;
a2 = -1700. - 0.099*T;
a3 = -35534. + 47.534*T;
a4 = 139840. - 97.424*T;
a5 = -65400. + 48.392*T;


ret = a0 + (a1 + R*T*log(xcu./(1-xcu))) .* xcu + a2 * xcu.^2 + a3 * xcu.^3 + a4 * xcu.^4 + a5 * xcu.^5 + R*T*log(1-xcu);

