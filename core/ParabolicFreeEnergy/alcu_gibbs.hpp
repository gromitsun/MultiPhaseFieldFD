//
//  alcu_gibbs.hpp
//  SteinbachPhaseFieldFD
//
//  Created by Yue Sun on 2/20/16.
//  Copyright (c) 2016 Yue Sun. All rights reserved.
//

#ifndef __SteinbachPhaseFieldFD__alcu_gibbs__
#define __SteinbachPhaseFieldFD__alcu_gibbs__


#include <cmath>

using namespace std;

/* Free energy functions */
/* x = X_cu (composition of Cu) */

#define R 8.314 // gas constant
#define Vma 9.9e-6 // molar volume of the alpha phase in m^3/mole
#define Vmb 9.9e-6 // molar volume of the liquid phase in m^3/mole

template <typename Type>
inline Type sq(const Type x)
{
	return x*x;
}


template <typename Type>
inline Type cu(const Type x)
{
	return x*x*x;
}


template <typename Type>
inline Type fa(const Type x, const Type T) // AlCu alpha
{
    Type a0 = -11276.2 + 74092./T + 0.018532*sq(T) - 5.76423e-6*cu(T) + (223.048 - 38.5844*log(T))*T;
    Type a1 = -10254.2 - 21614./T - 0.0211888*sq(T) + 5.89345e-6*cu(T) + (-92.5632 + 14.472*log(T))*T;
    Type a2 = -68100. + 4.*T;
    Type a3 = 86540. - 4.*T;
    Type a4 = -4680.0;
    return a0
    + (a1 + R*T*log(x/(1-x))) * x
    + a2 * sq(x)
    + a3 * cu(x)
    + a4 * sq(sq(x))
    + R*T*log(1-x);
}

template <typename Type>
inline Type fb(const Type x, const Type T) // AlCu liquid
{
    Type a0 = -271.195 + 74092./T + 0.018532*sq(T) - 5.76423e-6*cu(T) + 7.9337e-20*pow(T,7) + (211.207 - 38.5844*log(T))*T;
    Type a1 = -31740.5 - 21614./T - 0.0211888*sq(T) + 5.89345e-6*cu(T) - 8.51859e-20*pow(T,7)
                + (-88.6363 + 14.472*log(T))*T;
    Type a2 = -1700. - 0.099*T;
    Type a3 = -35534. + 47.534*T;
    Type a4 = 139840. - 97.424*T;
    Type a5 = -65400. + 48.392*T;
    
    return a0
    + (a1 + R*T*log(x/(1-x))) * x
    + a2 * sq(x)
    + a3 * cu(x)
    + a4 * sq(sq(x))
    + a5 * sq(x)*cu(x)
    + R*T*log(1-x);
}


template <typename Type>
inline Type f1a(const Type x, const Type T) // AlCu alpha
{
    Type a1 = -10254.2 - 21614./T - 0.0211888*sq(T) + 5.89345e-6*cu(T) + (-92.5632 + 14.472*log(T))*T;
    Type a2 = -68100. + 4.*T;
    Type a3 = 86540. - 4.*T;
    Type a4 = -4680.0;
    return a1 + R*T*log(x/(1-x))
    + 2*a2 * x
    + 3*a3 * sq(x)
    + 4*a4 * cu(x);
}


template <typename Type>
inline Type f1b(const Type x, const Type T) // AlCu liquid
{
    Type a1 = -31740.5 - 21614./T - 0.0211888*sq(T) + 5.89345e-6*cu(T) - 8.51859e-20*pow(T,7)
    + (-88.6363 + 14.472*log(T))*T;
    Type a2 = -1700. - 0.099*T;
    Type a3 = -35534. + 47.534*T;
    Type a4 = 139840. - 97.424*T;
    Type a5 = -65400. + 48.392*T;
    
    return a1
    + R*T*log(x/(1-x))
    + 2*a2 * x
    + 3*a3 * sq(x)
    + 4*a4 * cu(x)
    + 5*a5 * sq(sq(x));
}


template <typename Type>
inline Type f2a(const Type x, const Type T) // AlCu alpha
{
    Type a2 = -68100. + 4.*T;
    Type a3 = 86540. - 4.*T;
    Type a4 = -4680.0;
    return R*T/(x*(1-x))
    + 2*a2
    + 6*a3 * x
    + 12*a4 * sq(x);
}


template <typename Type>
inline Type f2b(const Type x, const Type T) // AlCu liquid
{
    Type a2 = -1700. - 0.099*T;
    Type a3 = -35534. + 47.534*T;
    Type a4 = 139840. - 97.424*T;
    Type a5 = -65400. + 48.392*T;
    
    return R*T/(x*(1-x))
    + 2*a2
    + 6*a3 * x
    + 12*a4 * sq(x)
    + 20*a5 * cu(x);
}



/* Other functions */
// approximate equilibrium compositions at a given temperature T
// from polynomial fitting of the phase diagram
template <typename Type>
inline void calc_compeq(const Type T, Type & compa_eq, Type & compb_eq)
{
    /* fitting parameters */
    const double a2 = 8.130346e-07;
    const double a1 = -1.642576e-03;
    const double a0 = 8.253158e-01;
    
    const double b2 = -1.914544e-06;
    const double b1 = 1.792344e-03;
    const double b0 = -6.086896e-03;
    
    compa_eq = (a2 * T + a1) * T + a0;
    compb_eq = (b2 * T + b1) * T + b0;
    
}


// approximate parabolic coefficents of the free energy functions 
// near equilibrium compositions at a given temperature T
template <typename Type>
inline void calc_parabolic(const Type T, const Type compa_eq, const Type compb_eq,
                           Type & a2, Type & a1, Type & a0,
                           Type & b2, Type & b1, Type & b0)
{
    a2 = f2a(compa_eq, T) / Vma;
    a1 = f1a(compa_eq, T) / Vma - 2*a2*compa_eq;
    a0 = fa(compa_eq, T) / Vma - (a2*compa_eq + a1) * compa_eq;
    
    b2 = f2b(compb_eq, T) / Vmb;
    b1 = f1b(compb_eq, T) / Vmb - 2*b2*compb_eq;
    b0 = fb(compb_eq, T) / Vmb - (b2*compb_eq + b1) * compb_eq;
}





// inline Type fa(const Type xcu, const Type T) // AlCu alpha
// {
//     //Aluminum References
//     Type GHSERAL = -11276.24+223.048446*T-38.5844296*T*log(T)+18.531982E-3*sq(T)-5.764227E-6*cu(T)+74092.0/T; //Valid up to 1234 K

//     //Copper References
//     Type GHSERCU = -7770.458+130.485235*T-24.112392*T*log(T)-2.65684E-3*sq(T)+0.129223E-6*cu(T)+52478.0/T; //Valid up to 1357 K

//     //Composition
//     Type xal=1-xcu;

//     Type gal=GHSERAL;
//     Type gcu=GHSERCU;

//     Type g_mix=(gal)*xal+(gcu)*xcu;

//     Type g_reg=R*T*(xal*log(xal)+xcu*log(xcu));

//     // Al-Cu Liquid
//     Type L0_alcu=(-53520+2*T);
//     Type L1_alcu=(38590-2*T);
//     Type L2_alcu=(1170);

//     Type g_alcu=xal*xcu*(L0_alcu+(xal-xcu)*L1_alcu+(xal-xcu)**2*L2_alcu);


//     Type g_ex=g_alcu; //+g_agsn+g_gcusn

//     return g_mix+g_reg+g_ex;
// }


// inline Type fb(const Type xcu, const Type T) // AlCu liquid
// {
//     //Al References
//     Type GHSERAL =  -11276.24+223.048446*T-38.5844296*T*log(T)+18.531982E-3*sq(T)-5.764227E-6*cu(T)+74092.0/T; //Valid up to 1234 K
//     Type GLIQAL =  11005.045-11.84185*T+79.337E-21*T**7+GHSERAL; //Valid up to 1234 K

//     //Copper References
//     Type GHSERCU =  -7770.458+130.485235*T-24.112392*T*log(T)-2.65684E-3*sq(T)+0.129223E-6*cu(T)+52478.0/T; //Valid up to 1357 K
//     Type GLIQCU = 12964.735-9.511904*T-584.89E-23*pow(T,7)+GHSERCU; //Valid up to 1357 K

//     //Composition
//     Type xal=1-xcu;

//     Type gal=GLIQAL;
//     Type gcu=GLIQCU;

//     Type g_mix=gal*xal+gcu*xcu;


//     Type g_reg=R*T*(xal*log(xal)+xcu*log(xcu)); //xcu*log(xcu)

//     // Al-Cu Liquid from Table 6 in [04Wit]
//     Type L0_alcu=-67094+8.555*T;
//     Type L1_alcu=32148-7.118*T;
//     Type L2_alcu=5915-5.889*T;
//     Type L3_alcu=-8175+6.049*T;

//     Type g_alcu=xal*xcu*(L0_alcu+(xal-xcu)*L1_alcu+(xal-xcu)**2*L2_alcu+(xal-xcu)**3*L3_alcu);

//     Type g_ex=g_alcu;

//     return g_mix+g_reg+g_ex;
// }


// inline Type f1a(const Type xcu, const Type T) // AlCu alpha
// {
//     return 0.350578E4+(-21614.0)/T+(-0.925632E2)*T+(-0.211888E-1)* \
//               sq(T)+0.589345E-5*cu(T)+((-53520)+2*T+(38590-2*T)*(1-2*xcu) \
//               +1170*sq(1+(-2)*xcu))*(1-xcu)+(-1)*((-53520)+2*T+(38590+( \
//               -2)*T)*(1+(-2)*xcu)+1170*(1+(-2)*xcu)**2)*xcu+((-2)*(38590-2* \
//               T)+(-4680)*(1-2*xcu))*(1-xcu)*xcu+0.14472E2*T*log(T)+ \
//               0.83145E1*T*((-1)*log(1-xcu)+log(xcu));
// }


// inline Type f2a(const Type xcu, const Type T) // AlCu alpha
// {
// 	return 107040+(-4)*T+0.83145E1*T*((1+(-1)*xcu)**(-1)+xcu**(-1))+(-2)*( \
// 			  38590+(-2)*T)*(1+(-2)*xcu)+(-2340)*(1+(-2)*xcu)**2+2*((-2)*( \
// 			  38590+(-2)*T)+(-4680)*(1+(-2)*xcu))*(1+(-1)*xcu)+(-2)*((-2)*( \
// 			  38590+(-2)*T)+(-4680)*(1+(-2)*xcu))*xcu+9360*(1+(-1)*xcu)*xcu;
// }

#endif