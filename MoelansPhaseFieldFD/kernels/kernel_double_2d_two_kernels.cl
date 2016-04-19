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

/* inline functions */
// calculate phase compositions from parabolic energy functions 
// by parallel tangent construction
inline void parabolic_comp(const double ha,
                           const double comp,
                           const double a2,
                           const double a1,
                           const double b2,
                           const double b1,
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


/**********************************************************************************/

/* kernels */

__kernel void step_phi_2d(// input/output arrays
                      __global double * Phi,
                      __global double * Comp,
                      __global double * PhiNext,
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
                      const double delta_comp_eq,
                      const double mk,
                      const double gamma,
                      const double kappa,
                      const double Da,
                      const double Db,
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
    double comp = Comp[pos];
    
    // Calculate h(phi)
    double ha = CalcH(phi);

    
    // Calculate parallel compositions
    double compa,  compb;
    parabolic_comp(ha, comp, a2, a1, b2, b1, &compa, &compb);
    
    // Compute M
    double m = CalcM(ha, a2, b2, Da, Db);
    
    // Compute U
    double u = a2 * compa + a1;

    // Write M and U to global memory
    U[pos] = u;
    M[pos] = m;
    
    // Step forward Phi
    double laplacian = (phi_xm+phi_xp+phi_ym+phi_yp-4.0*phi)/(dx*dx);  // 5-point stencil
    double L = 4 * mk * m / (3 * kappa * (delta_comp_eq*delta_comp_eq));
    // double L = 4 * mk * 0.5*(Da/a2+Db/b2) / (3 * kappa * (delta_comp_eq*delta_comp_eq));
    // Write PhiNext to memory object
    PhiNext[pos] = phi - dt * L * (mk * phi * (phi*phi - 1 + 2*gamma*(1-phi)*(1-phi))
                                      - kappa * laplacian
                                      + 2*phi/(phi*phi+(1-phi)*(1-phi))*(1-ha)*(CalcF(compa, a2, a1, a0)-CalcF(compb, b2, b1, b0)-u*(compa-compb)));
}


__kernel void step_comp_2d(__global double * Comp,
                           __global double * U,
                           __global double * M,
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
    
    // Read in Comp
    double comp = Comp[pos];
    
    // Read in U
    double u    = U[pos];
    double u_xm = U[pos-ILX];
    double u_xp = U[pos+ISX];
    double u_ym = U[pos-nx*ILY];
    double u_yp = U[pos+nx*ISY];

    // Read in M
    double m    = M[pos];
    double m_xm = M[pos-ILX];
    double m_xp = M[pos+ISX];
    double m_ym = M[pos-nx*ILY];
    double m_yp = M[pos+nx*ISY];

    // Step forward Comp
    double laplacian = ((m_xp+m)*(u_xp-u)
                        - (m+m_xm)*(u-u_xm) 
                        + (m_yp+m)*(u_yp-u) 
                        - (m+m_ym)*(u-u_ym)) / (2*dx*dx);
    // Write CompNext to memory object
    Comp[pos] = comp + dt * laplacian;
    
//    Comp[pos] = laplacian;
}