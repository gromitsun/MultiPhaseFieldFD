#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define __NINE_STENCIL__


inline double sq(const double x)
{
  return x*x;
}


inline double cu(const double x)
{
  return x*x*x;
}

/**********************************************************************************/


/* Macros */

#define GetPos2(x,y,NX) (NX*(y)+(x))
#define GetPos3(x,y,z,NX,NY) (NX*NY*(z)+NX*(y)+(x))

#define CalcH(phi) ((phi)*(phi)/((1-phi)*(1-phi)+(phi)*(phi)));
#define CalcM(h, a2, b2, Da, Db) ((h)*(Da)/(a2) + (1-(h))*(Db)/(b2))
#define CalcF(x, a2, a1, a0) (0.5*(a2)*(x)*(x)+a1*(x)+a0)

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
    double tmp1 = 1.0/(b2*ha+a2*(1-ha));
    double tmp2 = a1-b1;
    *compa = (b2*comp - (1-ha) * tmp2)*tmp1;
    *compb = (a2*comp +    ha  * tmp2)*tmp1;
}


inline double calc_ha(const double phi)
{
    return sq(phi)/(sq(1-phi)+sq(phi));
}


inline double calc_m(const double ha,
                     const double a2,
                     const double b2,
                     const double Da,
                     const double Db)
{
    return ha*Da/a2 + (1-ha)*Db/b2;
}


__kernel void step_2d(// input/output arrays
                      __global double * Phi,
                      __global double * Comp,
                      __global double * PhiNext,
                      __global double * CompNext,
                      // parabolic coefficients of the free energy functions
                      const double a2,
                      const double a1,
                      const double a0,
                      const double b2,
                      const double b1,
                      const double b0,
                      // physical parameters
                      const double mk,
                      const double gamma,
                      const double kappa,
                      const double Da,
                      const double Db,
                      const double delta_comp_eq,
                      // simulation parameters
                      const double dx,
                      const double dt)
{
    // Get coordinates and size
//    uint x = get_global_id(0);
//    uint y = get_global_id(1);
    uint nx = get_global_size(0);
//    uint ny = get_global_size(1);
    uint pos = GetPos2(get_global_id(0),get_global_id(1),nx);
    
    // Read in Phi
    double phi    = Phi[pos];
    // Calculate stencils
    double phi_xm = Phi[pos-ILX];
    double phi_xp = Phi[pos+ISX];
    double phi_ym = Phi[pos-nx*ILY];
    double phi_yp = Phi[pos+nx*ISY];
    
    // Read in Comp
    double comp    = Comp[pos];
    // Calculate stencils
    double comp_xm = Comp[pos-ILX];
    double comp_xp = Comp[pos+ISX];
    double comp_ym = Comp[pos-nx*ILY];
    double comp_yp = Comp[pos+nx*ISY];
    
    // Calculate h(phi)
    double ha    = CalcH(phi   );
    double ha_xm = CalcH(phi_xm);
    double ha_xp = CalcH(phi_xp);
    double ha_ym = CalcH(phi_ym);
    double ha_yp = CalcH(phi_yp);

    
    // Calculate parallel compositions
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
    double m    = CalcM(ha,    a2, b2, Da, Db);
    double m_xp = CalcM(ha_xp, a2, b2, Da, Db);
    double m_xm = CalcM(ha_xm, a2, b2, Da, Db);
    double m_yp = CalcM(ha_yp, a2, b2, Da, Db);
    double m_ym = CalcM(ha_ym, a2, b2, Da, Db);
    
    // Compute U
    double u    = a2 * compa    + a1;
    double u_xp = a2 * compa_xp + a1;
    double u_xm = a2 * compa_xm + a1;
    double u_yp = a2 * compa_yp + a1;
    double u_ym = a2 * compa_ym + a1;
    
    // Step forward Comp
    double laplacian = ((m_xp+m)*(u_xp-u)
                        - (m+m_xm)*(u-u_xm) 
                        + (m_yp+m)*(u_yp-u) 
                        - (m+m_ym)*(u-u_ym)) / (2*dx*dx);
    // Write CompNext to memory object
    CompNext[pos] = comp + dt * laplacian;
    
    
    // Step forward Phi
    laplacian = (phi_xm+phi_xp+phi_ym+phi_yp-4.0*phi)/(dx*dx);  // 5-point stencil
    double L = 4 * mk * m / (3 * kappa * (delta_comp_eq*delta_comp_eq));
    // Write PhiNext to memory object
    PhiNext[pos] = phi - dt * L * (mk * phi * (phi*phi - 1 + 2*gamma*(1-phi)*(1-phi))
                                      - kappa * laplacian
                                      + 2*phi/(phi*phi+(1-phi)*(1-phi))*(1-ha)*(CalcF(compa, a2, a1, a0)-CalcF(compb, b2, b1, b0)-u*(compa-compb)));
}