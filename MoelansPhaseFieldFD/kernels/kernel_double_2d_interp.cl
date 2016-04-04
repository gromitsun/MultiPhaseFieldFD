#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define __NINE_STENCIL__


/* Free energy functions */
#include "alcu_gibbs.hpp"

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

inline double calc_ha(const double phi)
{
    return phi*phi/(phi*phi+(1-phi)*(1-phi));
}


inline double interp_3d(__global const double * A,
                        const double x,
                        const double y,
                        const double z,
                        const size_t nx,
                        const size_t ny)
{
    int x0 = (int) x;
    int y0 = (int) y;
    int z0 = (int) z;
    
    int x1 = x0+1;
    int y1 = y0+1;
    int z1 = z0+1;
    
    // interpolate x
    double c00 = A[GetPos3(x0,y0,z0,nx,ny)]*(x1-x) + A[GetPos3(x1,y0,z0,nx,ny)]*(x-x0);
    double c01 = A[GetPos3(x0,y0,z1,nx,ny)]*(x1-x) + A[GetPos3(x1,y0,z1,nx,ny)]*(x-x0);
    double c10 = A[GetPos3(x0,y1,z0,nx,ny)]*(x1-x) + A[GetPos3(x1,y1,z0,nx,ny)]*(x-x0);
    double c11 = A[GetPos3(x0,y1,z1,nx,ny)]*(x1-x) + A[GetPos3(x1,y1,z1,nx,ny)]*(x-x0);
    
    // interpolate y
    double c0 = c00*(y1-y) + c10*(y-y0);
    double c1 = c01*(y1-y) + c11*(y-y0);
    
    // interpolate z
    double c = c0*(z1-z) + c1*(z-z0);
    
    return c;
    
}


inline double interp_1d(__global const double * A,
                        const double x)
{
    int x0 = (int) x;
    int x1 = x0+1;
    return A[x0]*(x1-x) + A[x1]*(x-x0);
    
}


inline double interp_compa(__global const double * CompA,
                           const double ha,
                           const double comp,
                           const double T,
                           const double PHI_MIN,
                           const double COMP_MIN,
                           const double T_MIN,
                           const double PHI_INC,
                           const double COMP_INC,
                           const double T_INC,
                           const double PHI_NUM,
                           const double COMP_NUM)
{
    double x = (ha-PHI_MIN)/PHI_INC;
    double y = (comp-COMP_MIN)/COMP_INC;
    double z = (T-T_MIN)/T_INC;
    
    return interp_3d(CompA, x, y, z, PHI_NUM, COMP_NUM);
}


inline double calc_m(const double ha,
                     const double comp,
                     const double compa,
                     const double T,
                     const double Da,
                     const double Db)
{
    return ha*Da/f2a(compa, T) + (1-ha)*Db/f2b(_COMPB_, T);
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
    double compa = interp_compa(CompA, ha, comp, T, PHI_MIN, COMP_MIN, T_MIN, PHI_INC, COMP_INC, T_INC, PHI_NUM, COMP_NUM);
    // Calculate stencils
    double compa_xp = interp_compa(CompA, ha_xp, comp_xp, T, PHI_MIN, COMP_MIN, T_MIN, PHI_INC, COMP_INC, T_INC, PHI_NUM, COMP_NUM);
    double compa_xm = interp_compa(CompA, ha_xm, comp_xm, T, PHI_MIN, COMP_MIN, T_MIN, PHI_INC, COMP_INC, T_INC, PHI_NUM, COMP_NUM);
    double compa_yp = interp_compa(CompA, ha_yp, comp_yp, T, PHI_MIN, COMP_MIN, T_MIN, PHI_INC, COMP_INC, T_INC, PHI_NUM, COMP_NUM);
    double compa_ym = interp_compa(CompA, ha_ym, comp_ym, T, PHI_MIN, COMP_MIN, T_MIN, PHI_INC, COMP_INC, T_INC, PHI_NUM, COMP_NUM);

    // Compute M
    double m = calc_m(ha, comp, compa, T, Da, Db);
    // Calculate stencils
    double m_xp = calc_m(ha_xp, comp_xp, compa_xp, T, Da, Db);
    double m_xm = calc_m(ha_xm, comp_xm, compa_xm, T, Da, Db);
    double m_yp = calc_m(ha_yp, comp_yp, compa_yp, T, Da, Db);
    double m_ym = calc_m(ha_ym, comp_ym, compa_ym, T, Da, Db);
    
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
    Comp[pos] = comp + dt * laplacian;

    
    
    // Read in DeltaCompEq
    double delta_comp_eq = interp_1d(DeltaCompEq, (T-T_MIN)/T_INC);

    
    // Step forward Phi
    laplacian = (phi_xm+phi_xp+phi_ym+phi_yp-4.0f*phi)/(dx*dx);  // 5-point stencil
    double L = 4*mk*m / (3 * kappa * (delta_comp_eq*delta_comp_eq));
    double compb = _COMPB_;
    // Write PhiNext to memory object
    Phi[pos] = phi - dt * L * (mk * phi * (phi*phi - 1 + 2*gamma*(1-phi)*(1-phi))
                                      - kappa * laplacian
                                      + 2*phi*(1-ha)*(fa(compa, T)-fb(compb, T)-u*(compa-compb)));

}