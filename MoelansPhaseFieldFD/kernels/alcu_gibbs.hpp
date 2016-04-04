//
//  alcu_gibbs.hpp
//  MoelansPhaseFieldFD
//
//  Created by Yue Sun on 2/20/16.
//  Copyright (c) 2016 Yue Sun. All rights reserved.
//

#ifndef __MoelansPhaseFieldFD__alcu_gibbs__
#define __MoelansPhaseFieldFD__alcu_gibbs__


/* Free energy functions */
/* x = X_cu (composition of Cu) */

#define R 8.314 // gas constant

inline double sq(const double x)
{
	return x*x;
}


inline double cu(const double x)
{
	return x*x*x;
}


inline double fa(const double x, const double T) // AlCu alpha
{
    double a0 = -11276.2 + 74092./T + 0.018532*sq(T) - 5.76423e-6*cu(T) + (223.048 - 38.5844*log(T))*T;
    double a1 = -10254.2 - 21614./T - 0.0211888*sq(T) + 5.89345e-6*cu(T) + (-92.5632 + 14.472*log(T))*T;
    double a2 = -68100. + 4.*T;
    double a3 = 86540. - 4.*T;
    double a4 = -4680.0;
    return a0
    + (a1 + R*T*log(x/(1-x))) * x
    + a2 * sq(x)
    + a3 * cu(x)
    + a4 * sq(sq(x))
    + R*T*log(1-x);
}

inline double fb(const double x, const double T) // AlCu liquid
{
    double a0 = -271.195 + 74092./T + 0.018532*sq(T) - 5.76423e-6*cu(T) + 7.9337e-20*pow(T,7) + (211.207 - 38.5844*log(T))*T;
    double a1 = -31740.5 - 21614./T - 0.0211888*sq(T) + 5.89345e-6*cu(T) - 8.51859e-20*pow(T,7)
                + (-88.6363 + 14.472*log(T))*T;
    double a2 = -1700. - 0.099*T;
    double a3 = -35534. + 47.534*T;
    double a4 = 139840. - 97.424*T;
    double a5 = -65400. + 48.392*T;
    
    return a0
    + (a1 + R*T*log(x/(1-x))) * x
    + a2 * sq(x)
    + a3 * cu(x)
    + a4 * sq(sq(x))
    + a5 * sq(x)*cu(x)
    + R*T*log(1-x);
}


inline double f1a(const double x, const double T) // AlCu alpha
{
    double a1 = -10254.2 - 21614./T - 0.0211888*sq(T) + 5.89345e-6*cu(T) + (-92.5632 + 14.472*log(T))*T;
    double a2 = -68100. + 4.*T;
    double a3 = 86540. - 4.*T;
    double a4 = -4680.0;
    return a1 + R*T*log(x/(1-x))
    + 2*a2 * x
    + 3*a3 * sq(x)
    + 4*a4 * cu(x);
}


inline double f2a(const double x, const double T) // AlCu alpha
{
    double a2 = -68100. + 4.*T;
    double a3 = 86540. - 4.*T;
    double a4 = -4680.0;
    return R*T/(x*(1-x))
    + 2*a2
    + 6*a3 * x
    + 12*a4 * sq(x);
}


inline double f2b(const double x, const double T) // AlCu liquid
{
    double a2 = -1700. - 0.099*T;
    double a3 = -35534. + 47.534*T;
    double a4 = 139840. - 97.424*T;
    double a5 = -65400. + 48.392*T;
    
    return R*T/(x*(1-x))
    + 2*a2
    + 6*a3 * x
    + 12*a4 * sq(x)
    + 20*a5 * cu(x);
}



// inline double fa(const double xcu, const double T) // AlCu alpha
// {
//     //Aluminum References
//     double GHSERAL = -11276.24+223.048446*T-38.5844296*T*log(T)+18.531982E-3*sq(T)-5.764227E-6*cu(T)+74092.0/T; //Valid up to 1234 K

//     //Copper References
//     double GHSERCU = -7770.458+130.485235*T-24.112392*T*log(T)-2.65684E-3*sq(T)+0.129223E-6*cu(T)+52478.0/T; //Valid up to 1357 K

//     //Composition
//     double xal=1-xcu;

//     double gal=GHSERAL;
//     double gcu=GHSERCU;

//     double g_mix=(gal)*xal+(gcu)*xcu;

//     double g_reg=R*T*(xal*log(xal)+xcu*log(xcu));

//     // Al-Cu Liquid
//     double L0_alcu=(-53520+2*T);
//     double L1_alcu=(38590-2*T);
//     double L2_alcu=(1170);

//     double g_alcu=xal*xcu*(L0_alcu+(xal-xcu)*L1_alcu+(xal-xcu)**2*L2_alcu);


//     double g_ex=g_alcu; //+g_agsn+g_gcusn

//     return g_mix+g_reg+g_ex;
// }


// inline double fb(const double xcu, const double T) // AlCu liquid
// {
//     //Al References
//     double GHSERAL =  -11276.24+223.048446*T-38.5844296*T*log(T)+18.531982E-3*sq(T)-5.764227E-6*cu(T)+74092.0/T; //Valid up to 1234 K
//     double GLIQAL =  11005.045-11.84185*T+79.337E-21*T**7+GHSERAL; //Valid up to 1234 K

//     //Copper References
//     double GHSERCU =  -7770.458+130.485235*T-24.112392*T*log(T)-2.65684E-3*sq(T)+0.129223E-6*cu(T)+52478.0/T; //Valid up to 1357 K
//     double GLIQCU = 12964.735-9.511904*T-584.89E-23*pow(T,7)+GHSERCU; //Valid up to 1357 K

//     //Composition
//     double xal=1-xcu;

//     double gal=GLIQAL;
//     double gcu=GLIQCU;

//     double g_mix=gal*xal+gcu*xcu;


//     double g_reg=R*T*(xal*log(xal)+xcu*log(xcu)); //xcu*log(xcu)

//     // Al-Cu Liquid from Table 6 in [04Wit]
//     double L0_alcu=-67094+8.555*T;
//     double L1_alcu=32148-7.118*T;
//     double L2_alcu=5915-5.889*T;
//     double L3_alcu=-8175+6.049*T;

//     double g_alcu=xal*xcu*(L0_alcu+(xal-xcu)*L1_alcu+(xal-xcu)**2*L2_alcu+(xal-xcu)**3*L3_alcu);

//     double g_ex=g_alcu;

//     return g_mix+g_reg+g_ex;
// }


// inline double f1a(const double xcu, const double T) // AlCu alpha
// {
//     return 0.350578E4+(-21614.0)/T+(-0.925632E2)*T+(-0.211888E-1)* \
//               sq(T)+0.589345E-5*cu(T)+((-53520)+2*T+(38590-2*T)*(1-2*xcu) \
//               +1170*sq(1+(-2)*xcu))*(1-xcu)+(-1)*((-53520)+2*T+(38590+( \
//               -2)*T)*(1+(-2)*xcu)+1170*(1+(-2)*xcu)**2)*xcu+((-2)*(38590-2* \
//               T)+(-4680)*(1-2*xcu))*(1-xcu)*xcu+0.14472E2*T*log(T)+ \
//               0.83145E1*T*((-1)*log(1-xcu)+log(xcu));
// }


// inline double f2a(const double xcu, const double T) // AlCu alpha
// {
// 	return 107040+(-4)*T+0.83145E1*T*((1+(-1)*xcu)**(-1)+xcu**(-1))+(-2)*( \
// 			  38590+(-2)*T)*(1+(-2)*xcu)+(-2340)*(1+(-2)*xcu)**2+2*((-2)*( \
// 			  38590+(-2)*T)+(-4680)*(1+(-2)*xcu))*(1+(-1)*xcu)+(-2)*((-2)*( \
// 			  38590+(-2)*T)+(-4680)*(1+(-2)*xcu))*xcu+9360*(1+(-1)*xcu)*xcu;
// }

#endif