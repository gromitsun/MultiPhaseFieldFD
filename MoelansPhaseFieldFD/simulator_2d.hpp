//
//  simulator_2d.hpp
//  MoelansPhaseFieldFD
//
//  Created by Yue Sun on 2/20/16.
//  Copyright (c) 2016 Yue Sun. All rights reserved.
//

#ifndef __MoelansPhaseFieldFD__simulator_2d__
#define __MoelansPhaseFieldFD__simulator_2d__

#include "simulator.hpp"

template <typename Type>
struct Parameter
{
    /* Input parameters */
    // physical parameters
    Type Da;
    Type Db;
    Type sigma;
    Type l;
    Type T;
    Type dT_dt;
    Type T_start;
    // phase field parameters
    Type gamma;    // = 1.5
    Type kappa;    // = 0.75*sigma*l
    Type mk;       // = 6*sigma/l
    // simulation parameters
    Type dx;
    Type dt;
    unsigned int nt;
    unsigned int t_skip; // output every t_skip steps
    Type dT_recalc;
};

template <typename Type>
struct Variable
{
    // parabolic coefficients of the free energy functions
    Type a2;
    Type a1;
    Type a0;
    Type b2;
    Type b1;
    Type b0;
    // other physical variables
    Type T;
    Type T_gibbs;       // temperature used in Gibbs free energy functions
    Type T_gibbs_next;
    Type delta_comp_eq;
    Type compa_eq;
    Type compb_eq;
};

template <typename Type>
class Simulator_2D : public Simulator<Type>
{
private:
    /* Input parameters */
    Parameter<Type> _paras;

    /* Simulation variables */
    Variable<Type> _vars;
    
    /* Host arrays */
    Type * _Phi;
    Type * _Comp;
    
    
    /* OpenCL variables */
    
    // OpenCL memory objects
    cl_mem _mem_Phi;
    cl_mem _mem_Comp;
    cl_mem _mem_PhiNext;
    cl_mem _rotate_var;
    cl_mem _mem_U;
    cl_mem _mem_M;
    
    // OpenCL kernel
    cl_kernel _kernel_step_phi_2d;
    cl_kernel _kernel_step_comp_2d;
    
    // OpenCL sizes
    size_t _local_size[3];
    size_t _global_size[3];
    
public:
    Simulator_2D();
    Simulator_2D(const unsigned int nx,
                 const unsigned int ny);
    Simulator_2D(const unsigned int nx,
                 const unsigned int ny,
                 const Parameter<Type> & paras);
    ~Simulator_2D();
    
    /* read input from files */
    void read_input(const char * filename);                             // read input parameters
    void read_init_cond(const char * phi_file, const char * comp_file); // read initial condition from data

    /* initialization */
    cl_int build_kernel(const char * kernel_file="kernel_float.cl");    // build OpenCL kernel
    void init_sim(const Type mean, const Type sigma);                   // initialize arrays and mem objects
    
    /* data transfer between host & device */
    cl_int write_mem();     // write Phi and Comp to buffer     (host   -> device)
    cl_int read_mem();      // read Phi and Comp from buffer    (device -> host)
    
    /* simulation steps */
    void step(const Type dt); // do one step
    void steps(const Type dt, const unsigned int nsteps,
               const bool finish=true, const bool cputime=true); // do multiple steps
    void set_temp(const Type T); // set the temperature
    void set_temp(); // set the temperature
    
    void run(); // run simulation according to program

    void restart(const unsigned int t); // restart from #t steps; reset step counter to t
    
    /* write output file */
    void writefile();
    
};


#endif /* defined(__AllenCahnFD__simulator_2d__) */
