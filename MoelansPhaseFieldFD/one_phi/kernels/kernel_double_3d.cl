#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// #define __NINETEEN_STENCIL__


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

#define CalcH(phia) ((phia)*(phia)/((phib)*(phib)+(phia)*(phia)));
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

/* inline functions */
// calculate phase compositions from parabolic energy functions
// by parallel tangent construction
inline void parabolic_comp(const double ha,
                           const double hb,
                           const double comp,
                           const double a2,
                           const double a1,
                           const double b2,
                           const double b1,
                           double *compa,
                           double *compb)
{
    double tmp1 = 1.0/(b2*ha+a2*hb);
    double tmp2 = a1-b1;
    *compa = (b2*comp - hb * tmp2)*tmp1;
    *compb = (a2*comp + ha * tmp2)*tmp1;
}


inline double calc_ha(const double phia, const double phib)
{
    return sq(phia)/(sq(phib)+sq(phia));
}


inline double laplacian_3d(__global const double * Phi,
                           const double phi,
                           const double dx,
                           const uint pos)
{
    // Calculate stencils
    double xm = Phi[pos-ILX];
    double xp = Phi[pos+ISX];
    double ym = Phi[pos-get_global_size(0)*ILY];
    double yp = Phi[pos+get_global_size(0)*ISY];
    double zm = Phi[pos-get_global_size(0)*get_global_size(1)*ILZ];
    double zp = Phi[pos+get_global_size(0)*get_global_size(1)*ISZ];
#ifndef __NINETEEN_STENCIL__
    return (xm+xp+ym+yp+zm+zp-6.0f*phi)/(dx*dx);  // 7-point stencil
#else
    double xym = Phi[pos-ILX-get_global_size(0)*ILY];
    double xyp = Phi[pos+ISX+get_global_size(0)*ISY];
    double xpym = Phi[pos+ILX-get_global_size(0)*ILY];
    double xmyp = Phi[pos-ILX+get_global_size(0)*ILY];
    
    return ((xm+xp+ym+yp+zm+zp)/3.0f
            + (xyp+xym+xpym+xmyp+xzm+xzp+xmzp+xpzm+yzm+yzp+ymzp+ypzm)/6.0f
            - 4.0f*phi)/(dx*dx); // 19-point stencil
#undef __NINETEEN_STENCIL__
#endif
}


/**********************************************************************************/

/* kernels */

__kernel void step_phi_3d(// input/output arrays
                          __global double * PhiA,
                          __global double * Comp,
                          __global double * PhiANext,
                          __global double * U,
                          __global double * M,
                          // parabolic coefficients of the free energy functions
                          const double a2,
                          const double a1,
                          const double a0,
                          const double b2,
                          const double b1,
                          const double b0,
                          // physical parameters
                          const double L,
                          const double Da,
                          const double Db,
                          const double mk,
                          const double gamma,
                          const double kappa,
                          // simulation parameters
                          const double dx,
                          const double dt)
{
    // Get coordinates and size
    uint pos = GetPos3(get_global_id(0),get_global_id(1),get_global_id(2),get_global_size(0),get_global_size(1));
    
    // Read in Phi
    double phia = PhiA[pos];
    
    // Read in Comp
    double comp = Comp[pos];
    
    // Calculate h(phia)
    double phia_sq = sq(phia);
    double phib_sq = sq(1-phia);
    double ha = phia_sq/(phia_sq + phib_sq);
    double hb = 1-ha;
    
    // Calculate parallel compositions
    double compa,  compb;
    parabolic_comp(ha, hb, comp, a2, a1, b2, b1, &compa, &compb);
    
    // Compute U & M
    double u = a2 * compa + a1;
    double m = ha * Da/a2 + hb * Db/b2;
    
    // Write U & M to global memory
    U[pos] = u;
    M[pos] = m;
    
    // Step forward Phi
    double laplacian = laplacian_3d(PhiA, phia, dx, pos);
    double temp = 2.0 / (phia_sq + phib_sq) * (CalcF(compa, a2, a1, a0)-CalcF(compb, b2, b1, b0)-u*(compa-compb));
    
    
    // Write PhiNext to memory object
    PhiANext[pos] = phia - dt * L * (mk * phia * (phia_sq - 1 + 2 * gamma * phib_sq)
                                     - kappa * laplacian
                                     + phia * hb * temp);
}


__kernel void step_comp_3d(__global double * Comp,
                           __global double * U,
                           __global double * M,
                           // simulation parameters
                           const double dx,
                           const double dt)
{
    // Get coordinates and size
    uint pos = GetPos3(get_global_id(0),get_global_id(1),get_global_id(2),get_global_size(0),get_global_size(1));
    
    // Read in Comp
    double comp = Comp[pos];
    
    // Read in U
    double u    = U[pos];
    double u_xm = U[pos-ILX];
    double u_xp = U[pos+ISX];
    double u_ym = U[pos-get_global_size(0)*ILY];
    double u_yp = U[pos+get_global_size(0)*ISY];
    double u_zm = U[pos-get_global_size(0)*get_global_size(1)*ILZ];
    double u_zp = U[pos+get_global_size(0)*get_global_size(1)*ISZ];
    
    // Read in M
    double m    = M[pos];
    double m_xm = M[pos-ILX];
    double m_xp = M[pos+ISX];
    double m_ym = M[pos-get_global_size(0)*ILY];
    double m_yp = M[pos+get_global_size(0)*ISY];
    double m_zm = M[pos-get_global_size(0)*get_global_size(1)*ILZ];
    double m_zp = M[pos+get_global_size(0)*get_global_size(1)*ISZ];
    
    // Step forward Comp
    double laplacian = ((m_xp+m)*(u_xp-u)
                        - (m+m_xm)*(u-u_xm)
                        + (m_yp+m)*(u_yp-u)
                        - (m+m_ym)*(u-u_ym)
                        + (m_zp+m)*(u_zp-u)
                        - (m+m_zm)*(u-u_zm)) / (2*dx*dx);
    
    // Write CompNext to memory object
    Comp[pos] = comp + dt * laplacian;
    
}