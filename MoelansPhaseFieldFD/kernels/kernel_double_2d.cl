#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define __NINE_STENCIL__


/* Free energy functions */
/* x = X_cu (composition of Cu) */

#define R 8.314 // gas constant

/* constants */
__constant double a1 = 5.562064123036832e+09;
__constant double a2 = 1.019435109224830e+10;
__constant double cS0 = 0.7821722753190940;
__constant double cL0 = 0.5704079007319450;
__constant double CS = -4.612094741994919e+09;
__constant double CL = -4.448563405669029e+09;
__constant double b1 = 0.7821722753190940*5.562064123036832e+09;
__constant double b2 = 0.5704079007319450*1.019435109224830e+10;
__constant double cSeq = 0.019862472879877200;
__constant double cLeq = 0.1544897158058190;

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
    return 0.5*a1*(x-cS0)*(x-cS0)+CS;
}

inline double fb(const double x, const double T) // AlCu liquid
{
    return 0.5*a2*(x-cL0)*(x-cL0)+CL;
}


inline double f1a(const double x, const double T) // AlCu alpha
{
    return x*a1+b1;
}


inline double f2a(const double x, const double T) // AlCu alpha
{
    return a1;
}


inline double f2b(const double x, const double T) // AlCu liquid
{
    return a2;
}

/**********************************************************************************/


/* Macros */

#define GetPos2(x,y,NX) (NX*(y)+(x))
#define GetPos3(x,y,z,NX,NY) (NX*NY*(z)+NX*(y)+(x))

#define _COMPB_ (ha==1 ? comp : (comp-ha*compa)/(1-ha))

//Macros to check if the current point is on the outside edges
#define ISX (get_global_id(0)<get_global_size(0)-1)
#define ISY (get_global_id(1)<get_global_size(1)-1)
#define ISZ (get_global_id(2)<get_global_size(2)-1)
#define ILX (get_global_id(0)>0)
#define ILY (get_global_id(1)>0)
#define ILZ (get_global_id(2)>0)

//#define ISX (x<nx-1)
//#define ISY (y<ny-1)
//#define ISZ (z<nz-1)
//#define ILX (x>0)
//#define ILY (y>0)
//#define ILZ (z>0)


/**********************************************************************************/

inline void parabolic_comp(const double ha,
                           const double comp,
                           double *compa,
                           double *compb)
{
    double tmp1 = 1.0/(a1*ha+a2*(1-ha));
    double tmp2 = b1-b2;
    *compa = (a2*comp + (1-ha) * tmp2)*tmp1;
    *compb = (a1*comp -    ha  * tmp2)*tmp1;
    
}


inline double calc_m1(const double ha,
                      const double comp,
                      const double compa,
                      const double compb,
                      const double T,
                      const double Da,
                      const double Db)
{
    return ha*Da/f2a(compa, T) + (1-ha)*Db/f2b(compb, T);
}


__kernel void step_2d(// input/output arrays
                      __global double * Phi,
                      __global double * Comp,
                      __global const double * CompA,
                      __global const double * DeltaCompEq,
                      // physical parameters
                      const double T,
                      const double mk,
                      const double gamma,
                      const double kappa,
                      const double Da,
                      const double Db,
                      // simulation parameters
                      const double dx,
                      const double dt,
                      // interpolation data info
                      const double PHI_MIN,
                      const double COMP_MIN,
                      const double T_MIN,
                      const double PHI_INC,
                      const double COMP_INC,
                      const double T_INC,
                      const double PHI_NUM,
                      const double COMP_NUM)
{
    // Get coordinates and size
//    uint x = get_global_id(0);
//    uint y = get_global_id(1);
    uint nx = get_global_size(0);
//    uint ny = get_global_size(1);
    uint pos = GetPos2(get_global_id(0),get_global_id(1),nx);
    
    // Read in Phi
    double phi = Phi[pos];
    // Calculate stencils
    double phi_xm = Phi[pos-ILX];
    double phi_xp = Phi[pos+ISX];
    double phi_ym = Phi[pos-nx*ILY];
    double phi_yp = Phi[pos+nx*ISY];
    
    // Read in Comp
    double comp = Comp[pos];
    // Calculate stencils
    double comp_xm = Comp[pos-ILX];
    double comp_xp = Comp[pos+ISX];
    double comp_ym = Comp[pos-nx*ILY];
    double comp_yp = Comp[pos+nx*ISY];
    
    // Calculate h(phi)
    double ha = calc_ha(phi);
    double ha_xm = calc_ha(phi_xm);
    double ha_xp = calc_ha(phi_xp);
    double ha_ym = calc_ha(phi_ym);
    double ha_yp = calc_ha(phi_yp);

    
    // Interpolate CompA
    double compa   ,  compb   ;
    double compa_xp,  compb_xp;
    double compa_xm,  compb_xm;
    double compa_yp,  compb_yp;
    double compa_ym,  compb_ym;
    parabolic_comp(ha,    comp,    &compa   , &compb   );
    parabolic_comp(ha_xp, comp_xp, &compa_xp, &compb_xp);
    parabolic_comp(ha_xm, comp_xm, &compa_xm, &compb_xm);
    parabolic_comp(ha_yp, comp_yp, &compa_yp, &compb_yp);
    parabolic_comp(ha_ym, comp_ym, &compa_ym, &compb_ym);
    
    // Compute M
    double m    = calc_m1(ha,    comp,    compa   , compa   , T, Da, Db);
    double m_xp = calc_m1(ha_xp, comp_xp, compa_xp, compa_xp, T, Da, Db);
    double m_xm = calc_m1(ha_xm, comp_xm, compa_xm, compa_xm, T, Da, Db);
    double m_yp = calc_m1(ha_yp, comp_yp, compa_yp, compa_yp, T, Da, Db);
    double m_ym = calc_m1(ha_ym, comp_ym, compa_ym, compa_ym, T, Da, Db);
    
    // Compute U
    double u = f1a(compa, T);
    // Calculate stencils
    double u_xp = f1a(compa_xp, T);
    double u_xm = f1a(compa_xm, T);
    double u_yp = f1a(compa_yp, T);
    double u_ym = f1a(compa_ym, T);
    
    // Step forward Comp
    double laplacian = ((m_xp+m)*(u_xp-u)
                        - (m+m_xm)*(u-u_xm) 
                        + (m_yp+m)*(u_yp-u) 
                        - (m+m_ym)*(u-u_ym)) / (2*dx*dx);
    // Write CompNext to memory object
    Comp[pos] = compb + dt * laplacian;
    
    
    // Read in DeltaCompEq
    double delta_comp_eq = cSeq - cLeq;
    
    
    // Step forward Phi
    laplacian = (phi_xm+phi_xp+phi_ym+phi_yp-4.0*phi)/(dx*dx);  // 5-point stencil
    double L = 4*mk / (3 * kappa * (delta_comp_eq*delta_comp_eq / m));
    // Write PhiNext to memory object
    Phi[pos] = phi - dt * L * (mk * (phi*phi - phi + 2*gamma*(1-phi)*(1-phi))
                                      - kappa * laplacian
                                      + 2*phi*(1-ha)*(fa(compa, T)-fb(compb, T)-u*(compa-compb)));
}