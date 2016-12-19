#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// #define __NINE_STENCIL__


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

#define GetPos2(x,y,NX) ((NX)*(y)+(x))
#define GetPos3(x,y,z,NX,NY) ((NX)*(NY)*(z)+(NX)*(y)+(x))

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
// Assume fa(x) = 0.5*a2*x^2 + a1*x + a0
//        fb(x) = 0.5*b2*x^2 + b1*x + b0
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

// Calculate phase compositions from parabolic energy functions
// by extrapolation (c.f. Steinbach 2006 PRE)
inline void extrapolate_comp(const double ha,
                             const double hb,
                             const double comp,
                             const double f2a,
                             const double f2b,
                             const double compa_eq,
                             const double compb_eq,
                             double *compa,
                             double *compb)
{
    double kba = f2a/f2b;
    double temp1 = (compb_eq - kba*compa_eq);
    double temp2 = (ha + hb * kba);
    *compa = (comp     - hb*temp1) / temp2;
    *compb = (comp*kba + ha*temp1) / temp2;
}


inline double calc_ha(const double phia, const double phib)
{
    return sq(phia)/(sq(phib)+sq(phia));
}


inline double laplacian_2d(__global const double * Phi,
                          const double phi,
                          const double dx,
                          const uint pos)
{
    // Calculate stencils
    double xm = Phi[pos-ILX];
    double xp = Phi[pos+ISX];
    double ym = Phi[pos-get_global_size(0)*ILY];
    double yp = Phi[pos+get_global_size(0)*ISY];
#ifndef __NINE_STENCIL__
    return (xm+xp+ym+yp-4.0f*phi)/(dx*dx);  // 5-point stencil
#else
    double xym = Phi[pos-ILX-get_global_size(0)*ILY];
    double xyp = Phi[pos+ISX+get_global_size(0)*ISY];
    double xpym = Phi[pos+ILX-get_global_size(0)*ILY];
    double xmyp = Phi[pos-ILX+get_global_size(0)*ILY];
    
    return ((xm+xp+ym+yp)/2.0f + (xyp+xym+xmyp+xpym)/4.0f - 3.0f*phi)/(dx*dx); // 9-point stencil
#undef __NINE_STENCIL__
#endif
}


/**********************************************************************************/

/* kernels */

__kernel void step_phi_2d(// input/output arrays
                          __global double * PhiA,
                          __global double * Comp,
                          __global double * PhiANext,
                          __global double * U,
                          // parabolic coefficients of the free energy functions
                          const double a2,
                          const double a1,
                          const double a0,
                          const double b2,
                          const double b1,
                          const double b0,
                          // physical parameters
                          const double L,
                          const double mk,
                          const double gamma,
                          const double kappa,
                          // simulation parameters
                          const double dx,
                          const double dt)
{
    // Get coordinates and size
    uint pos = GetPos2(get_global_id(0),get_global_id(1),get_global_size(0));
    
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
    
    // Compute U
    double u = a2 * compa + a1;

    // Write U to global memory
    U[pos] = u;
    
    // Step forward Phi
    double laplacian = laplacian_2d(PhiA, phia, dx, pos);
    double temp = 2.0 * phia * (1 - phia) / sq(phia_sq + phib_sq) * (CalcF(compa, a2, a1, a0)-CalcF(compb, b2, b1, b0)-u*(compa-compb));

    
    // Write PhiNext to memory object
//    PhiANext[pos] = phia - dt * L * (mk * phia * (phia_sq - 1 + 2 * gamma * phib_sq)
//                                      - kappa * laplacian
//                                      + phia * hb * temp);
    PhiANext[pos] = phia - dt * L * (mk * (1 + 2 * gamma) * phia * (1 - phia) * (1 - 2 * phia)
                                     - 2 * kappa * laplacian
                                     + temp);
}


__kernel void step_comp_2d(__global double * PhiA,
                           __global double * Comp,
                           __global double * U,
                           __local double * local_M,
                           // physical parameters
                           const double Ma,
                           const double Mb,
                           // simulation parameters
                           const double dx,
                           const double dt)
{
    // Get coordinates and size
    uint pos = GetPos2(get_global_id(0),get_global_id(1),get_global_size(0));
    uint local_pos = GetPos2(get_local_id(0)+1, get_local_id(1)+1, get_local_size(0)+2);
    uint local_line = get_local_size(0)+2;
    
    // Read in Phi
    double phia = PhiA[pos];
    
    // Read in Comp
    double comp = Comp[pos];
    
    // Read in U
    double u    = U[pos];
    double u_xm = U[pos-ILX];
    double u_xp = U[pos+ISX];
    double u_ym = U[pos-get_global_size(0)*ILY];
    double u_yp = U[pos+get_global_size(0)*ISY];

    // Calculate h(phia)
    double phia_sq = sq(phia);
    double phib_sq = sq(1-phia);
    double ha = phia_sq/(phia_sq + phib_sq);
    double hb = 1-ha;
    
    // Compute M
    double m = ha * Ma + hb * Mb;
    
    // Write M to local memory
    local_M[local_pos] = m;
    
    // Handle boundaries
    // x
    if (get_local_id(0) == 0)
    {
        // Read in Phi
        phia = PhiA[pos-ILX];
        // Calculate h(phia)
        phia_sq = sq(phia);
        phib_sq = sq(1-phia);
        ha = phia_sq/(phia_sq + phib_sq);
        hb = 1-ha;
        
        // Compute M
        m = ha * Ma + hb * Mb;
        
        // Write M to local memory
        local_M[local_pos-1] = m;
        
    }
    if (get_local_id(0) == get_local_size(0)-1)
    {
        // Read in Phi
        phia = PhiA[pos+ISX];
        // Calculate h(phia)
        phia_sq = sq(phia);
        phib_sq = sq(1-phia);
        ha = phia_sq/(phia_sq + phib_sq);
        hb = 1-ha;
        
        // Compute M
        m = ha * Ma + hb * Mb;
        
        // Write M to local memory
        local_M[local_pos+1] = m;
        
    }
    // y
    if (get_local_id(1) == 0)
    {
        // Read in Phi
        phia = PhiA[pos-get_global_size(0)*ILY];
        // Calculate h(phia)
        phia_sq = sq(phia);
        phib_sq = sq(1-phia);
        ha = phia_sq/(phia_sq + phib_sq);
        hb = 1-ha;
        
        // Compute M
        m = ha * Ma + hb * Mb;
        
        // Write M to local memory
        local_M[local_pos-local_line] = m;
        
    }
    if (get_local_id(1) == get_local_size(1)-1)
    {
        // Read in Phi
        phia = PhiA[pos+get_global_size(0)*ISY];
        // Calculate h(phia)
        phia_sq = sq(phia);
        phib_sq = sq(1-phia);
        ha = phia_sq/(phia_sq + phib_sq);
        hb = 1-ha;
        
        // Compute M
        m = ha * Ma + hb * Mb;
        
        // Write M to local memory
        local_M[local_pos+local_line] = m;
    }
    
    // Local memory barrier
    barrier(CLK_LOCAL_MEM_FENCE);
    

    // Read in M
    m = local_M[local_pos];
    double m_xm = local_M[local_pos-1];
    double m_xp = local_M[local_pos+1];
    double m_ym = local_M[local_pos-local_line];
    double m_yp = local_M[local_pos+local_line];

    // Step forward Comp
    double laplacian = ((m_xp+m)*(u_xp-u)
                        - (m+m_xm)*(u-u_xm) 
                        + (m_yp+m)*(u_yp-u) 
                        - (m+m_ym)*(u-u_ym)) / (2*dx*dx);
    
    // Write CompNext to memory object
    Comp[pos] = comp + dt * laplacian;
    
}