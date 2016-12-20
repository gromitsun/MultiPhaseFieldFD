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
                             const double kba,
                             const double compa_eq,
                             const double compb_eq,
                             double *compa,
                             double *compb)
{
    double temp1 = (compb_eq - kba*compa_eq);
    double temp2 = (ha + hb * kba);
    *compa = (comp     - hb*temp1) / temp2;
    *compb = (comp*kba + ha*temp1) / temp2;
}


// Evaluate free energy functions and their derivatives from coefficients
inline void eval_free_energy(const double a0,
                             const double a1,
                             const double a2,
                             const double a3,
                             const double a4,
                             const double b0,
                             const double b1,
                             const double b2,
                             const double b3,
                             const double b4,
                             const double b5,
                             const double compa,
                             const double compb,
                             const double RT,
                             double *fa,
                             double *fb,
                             double *f1a,
                             double *f1b,
                             double *f2a,
                             double *f2b)
{
    double x, sq_x, cu_x, qu_x, qe_x;
    double RT_log, RT_div, RT_log_1_x;
    
    x = compa;
    RT_log = RT * log(x/(1-x));
    RT_log_1_x = RT * log(1-x);
    RT_div = RT / (x*(1-x));
    sq_x = sq(x);
    cu_x = x*sq_x;
    qu_x = x*cu_x;
    *fa = a0
    + (a1 + RT_log) * x
    + a2 * sq_x
    + a3 * cu_x
    + a4 * qu_x
    + RT_log_1_x;
    *f1a = a1 + RT_log
    + 2*a2 * x
    + 3*a3 * sq_x
    + 4*a4 * cu_x;
    *f2a = RT_div
    + 2*a2
    + 6*a3 * x
    + 12*a4 * sq_x;
    
    x = compb;
    RT_log = RT * log(x/(1-x));
    RT_log_1_x = RT * log(1-x);
    RT_div = RT / (x*(1-x));
    sq_x = sq(x);
    cu_x = x*sq_x;
    qu_x = x*cu_x;
    qe_x = x*qu_x;
    *fb = b0
    + (b1 + RT_log) * x
    + b2 * sq_x
    + b3 * cu_x
    + b4 * qu_x
    + b5 * qe_x
    + RT_log_1_x;
    *f1b = b1 + RT_log
    + 2*b2 * x
    + 3*b3 * sq_x
    + 4*b4 * cu_x
    + 5*b5 * qu_x;
    *f2b = RT_div
    + 2*b2
    + 6*b3 * x
    + 12*b4 * sq_x
    + 20*b5 * cu_x;
}

// Evaluate free energy functions and their derivatives from coefficients (2nd derivatives only)
inline void eval_f1f2(const double a1,
                      const double a2,
                      const double a3,
                      const double a4,
                      const double b1,
                      const double b2,
                      const double b3,
                      const double b4,
                      const double b5,
                      const double compa,
                      const double compb,
                      const double RT,
                      double *f1a,
                      double *f1b,
                      double *f2a,
                      double *f2b)
{
    double x, sq_x, cu_x, qu_x;
    double RT_log, RT_div;
    
    x = compa;
    RT_log = RT * log(x/(1-x));
    RT_div = RT / (x*(1-x));
    sq_x = sq(x);
    cu_x = x*sq_x;
    *f1a = a1 + RT_log
    + 2*a2 * x
    + 3*a3 * sq_x
    + 4*a4 * cu_x;
    *f2a = RT_div
    + 2*a2
    + 6*a3 * x
    + 12*a4 * sq_x;
    
    x = compb;
    RT_log = RT * log(x/(1-x));
    RT_div = RT / (x*(1-x));
    sq_x = sq(x);
    cu_x = x*sq_x;
    qu_x = x*cu_x;
    *f1b = b1 + RT_log
    + 2*b2 * x
    + 3*b3 * sq_x
    + 4*b4 * cu_x
    + 5*b5 * qu_x;
    *f2b = RT_div
    + 2*b2
    + 6*b3 * x
    + 12*b4 * sq_x
    + 20*b5 * cu_x;
}


// Calculate laplacian by finite difference
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

__kernel void step_2d(// input/output arrays
                      __global double * PhiA,
                      __global double * Comp,
                      __global double * PhiANext,
                      __global double * CompNext,
                      __local double * local_U,
                      __local double * local_M,
                      // coefficients of the free energy functions
                      const double a0,
                      const double a1,
                      const double a2,
                      const double a3,
                      const double a4,
                      const double b0,
                      const double b1,
                      const double b2,
                      const double b3,
                      const double b4,
                      const double b5,
                      // R * T
                      const double RT,
                      // equilibrium compositions
                      const double compa_eq,
                      const double compb_eq,
                      // partition coefficient (f2a/f2b) at equilibrium compositions (c.f. Steinbach 2006 PRE)
                      const double kba,
                      // diffusion coefficients
                      const double Da,
                      const double Db,
                      // phase field parameters
                      const double L,
                      const double mk,
                      const double kappa,
                      // simulation parameters
                      const double dx,
                      const double dt)
{
    // Get coordinates and size
    uint pos = GetPos2(get_global_id(0),get_global_id(1),get_global_size(0));
    uint local_pos = GetPos3(get_local_id(0)+1, get_local_id(1)+1, get_local_id(2)+1, get_local_size(0)+2, get_local_size(1)+2);
    uint local_line = get_local_size(0)+2;
    
    // Read in Phi
    double phia = PhiA[pos];
    
    // Read in Comp
    double comp = Comp[pos];
    
    // Calculate h(phia)
    double phia_sq = sq(phia);
    double phib_sq = sq(1-phia);
    double ha = phia_sq/(phia_sq + phib_sq);
    double hb = 1-ha;
    
    /******* Compute phase compositions & energies *******/
    // Calculate parallel compositions
    double compa, compb;
    extrapolate_comp(ha, hb, comp, kba, compa_eq, compb_eq, &compa, &compb);
    
    // Evaluate free energy functions and derivatives
    double fa, fb, f1a, f1b, f2a, f2b;
    eval_free_energy(a0, a1, a2, a3, a4, b0, b1, b2, b3, b4, b5, compa, compb, RT, &fa, &fb, &f1a, &f1b, &f2a, &f2b);
    
    /******* Compute U & M *******/
    // Compute U & M
    double u = ha*f1a + hb*f1b;
    double m = ha*(Da/f2a) + hb*(Db/f2b);
    
    // Write U & M to local memory
    local_M[local_pos] = m;
    local_U[local_pos] = u;
    
    // Handle boundaries
    // x
    if (get_local_id(0) == 0)
    {
        // Read in Phi
        double phia = PhiA[pos-ILX];
        // Read in Comp
        double comp = Comp[pos-ILX];
        // Calculate h(phia)
        double phia_sq = sq(phia);
        double phib_sq = sq(1-phia);
        double ha = phia_sq/(phia_sq + phib_sq);
        double hb = 1-ha;
        
        // Calculate parallel compositions
        double compa, compb;
        extrapolate_comp(ha, hb, comp, kba, compa_eq, compb_eq, &compa, &compb);
        
        // Evaluate free energy functions 2nd derivatives
        double f1a, f1b, f2a, f2b;
        eval_f1f2(a1, a2, a3, a4, b1, b2, b3, b4, b5, compa, compb, RT, &f1a, &f1b, &f2a, &f2b);
        
        // Compute U & M
        double u = ha*f1a + hb*f1b;
        double m = ha*(Da/f2a) + hb*(Db/f2b);
        
        // Write U & M to local memory
        local_M[local_pos-1] = m;
        local_U[local_pos-1] = u;
        
    }
    if (get_local_id(0) == get_local_size(0)-1)
    {
        // Read in Phi
        double phia = PhiA[pos+ISX];
        // Read in Comp
        double comp = Comp[pos+ISX];
        // Calculate h(phia)
        double phia_sq = sq(phia);
        double phib_sq = sq(1-phia);
        double ha = phia_sq/(phia_sq + phib_sq);
        double hb = 1-ha;
        
        // Calculate parallel compositions
        double compa, compb;
        extrapolate_comp(ha, hb, comp, kba, compa_eq, compb_eq, &compa, &compb);
        
        // Evaluate free energy functions 2nd derivatives
        double f1a, f1b, f2a, f2b;
        eval_f1f2(a1, a2, a3, a4, b1, b2, b3, b4, b5, compa, compb, RT, &f1a, &f1b, &f2a, &f2b);
        
        // Compute U & M
        double u = ha*f1a + hb*f1b;
        double m = ha*(Da/f2a) + hb*(Db/f2b);
        
        // Write U & M to local memory
        local_M[local_pos+1] = m;
        local_U[local_pos+1] = u;
        
    }
    // y
    if (get_local_id(1) == 0)
    {
        // Read in Phi
        double phia = PhiA[pos-get_global_size(0)*ILY];
        // Read in Comp
        double comp = Comp[pos-get_global_size(0)*ILY];
        // Calculate h(phia)
        double phia_sq = sq(phia);
        double phib_sq = sq(1-phia);
        double ha = phia_sq/(phia_sq + phib_sq);
        double hb = 1-ha;
        
        // Calculate parallel compositions
        double compa, compb;
        extrapolate_comp(ha, hb, comp, kba, compa_eq, compb_eq, &compa, &compb);
        
        // Evaluate free energy functions 2nd derivatives
        double f1a, f1b, f2a, f2b;
        eval_f1f2(a1, a2, a3, a4, b1, b2, b3, b4, b5, compa, compb, RT, &f1a, &f1b, &f2a, &f2b);
        
        // Compute U & M
        double u = ha*f1a + hb*f1b;
        double m = ha*(Da/f2a) + hb*(Db/f2b);
        
        // Write U & M to local memory
        local_M[local_pos-local_line] = m;
        local_U[local_pos-local_line] = u;
        
    }
    if (get_local_id(1) == get_local_size(1)-1)
    {
        // Read in Phi
        double phia = PhiA[pos+get_global_size(0)*ISY];
        // Read in Comp
        double comp = Comp[pos+get_global_size(0)*ISY];
        // Calculate h(phia)
        double phia_sq = sq(phia);
        double phib_sq = sq(1-phia);
        double ha = phia_sq/(phia_sq + phib_sq);
        double hb = 1-ha;
        
        // Calculate parallel compositions
        double compa, compb;
        extrapolate_comp(ha, hb, comp, kba, compa_eq, compb_eq, &compa, &compb);
        
        // Evaluate free energy functions 2nd derivatives
        double f1a, f1b, f2a, f2b;
        eval_f1f2(a1, a2, a3, a4, b1, b2, b3, b4, b5, compa, compb, RT, &f1a, &f1b, &f2a, &f2b);
        
        // Compute U & M
        double u = ha*f1a + hb*f1b;
        double m = ha*(Da/f2a) + hb*(Db/f2b);
        
        // Write U & M to local memory
        local_M[local_pos+local_line] = m;
        local_U[local_pos+local_line] = u;
    }
    
    /******* Step forward PDE *******/
    // Step forward Phi
    double laplacian = laplacian_2d(PhiA, phia, dx, pos);
    double deltaG = 2.0 * (phia*hb + (1-phia)*ha) * (fb-fa-u*(compb-compa));
    
    phia = phia + dt * L * (mk * (2 * phia - 1) * (phia > 0) * (phia < 1)
                            + 2 * kappa * laplacian
                            + deltaG);
    
    // Write PhiNext to memory object
    PhiANext[pos] = (phia > 0) ? ((phia < 1) ? phia : 1) : 0;
    
    // Local memory barrier
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Read in U
    double u_xm = local_U[local_pos-1];
    double u_xp = local_U[local_pos+1];
    double u_ym = local_U[local_pos-local_line];
    double u_yp = local_U[local_pos+local_line];
    
    // Read in M
    double m_xm = local_M[local_pos-1];
    double m_xp = local_M[local_pos+1];
    double m_ym = local_M[local_pos-local_line];
    double m_yp = local_M[local_pos+local_line];
    
    // Step forward Comp
    laplacian = ((m_xp+m)*(u_xp-u)
                 - (m+m_xm)*(u-u_xm)
                 + (m_yp+m)*(u_yp-u)
                 - (m+m_ym)*(u-u_ym)) / (2*dx*dx);
    
    // Write CompNext to memory object
    CompNext[pos] = comp + dt * laplacian;
}