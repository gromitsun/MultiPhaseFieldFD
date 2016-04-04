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
class Simulator_2D : public Simulator<Type>
{
private:
    /* Input parameters */
    // physical parameters
    Type _Da;
    Type _Db;
    Type _sigma;
    Type _l;
    Type _T;
    Type _dT_dt;
    // phase field parameters
    Type _gamma;    // = 1.5
    Type _kappa;    // = 0.75*sigma*l
    Type _mk;       // = 6*sigma/l
    // simulation parameters
    Type _dx;
    Type _dt;
    unsigned int _nt;
    unsigned int _t_skip;
    // interpolation parameters
    Type _PHI_MIN;
    Type _COMP_MIN;
    Type _T_MIN;
    Type _PHI_INC;
    Type _COMP_INC;
    Type _T_INC;
    size_t _PHI_NUM;
    size_t _COMP_NUM;
    size_t _T_NUM;
    
    /* Host arrays */
    Type * _Phi;
    Type * _Comp;
    Type * _CompA;
    Type * _DeltaCompEq;
    
    
    /* OpenCL variables */
    
    // OpenCL memory objects
    cl_mem _mem_Phi;
    cl_mem _mem_Comp;
    cl_mem _mem_CompA;
    cl_mem _mem_DeltaCompEq;
    
    // OpenCL kernel
    cl_kernel _kernel_step_2d;
    
    // OpenCL sizes
    size_t _local_size[3];
    size_t _global_size[3];
    
public:
    Simulator_2D();
    Simulator_2D(const unsigned int nx,
                 const unsigned int ny);
    Simulator_2D(const unsigned int nx,
                 const unsigned int ny,
                 const Type dx,
                 const Type dt,
                 const Type Da,
                 const Type Db,
                 const Type sigma,
                 const Type l,
                 const Type T_start,
                 const Type dT_dt,
                 const unsigned int t_skip,
                 const Type PHI_MIN,
                 const Type COMP_MIN,
                 const Type T_MIN,
                 const Type PHI_INC,
                 const Type COMP_INC,
                 const Type T_INC,
                 const size_t PHI_NUM,
                 const size_t COMP_NUM,
                 const size_t T_NUM);
    ~Simulator_2D();
    
    /* read input from files */
    void read_input(const char * filename);                             // read input parameters
    void read_init_cond(const char * phi_file, const char * comp_file); // read initial condition from data
    void read_compa(const char * filename);                             // read paralell tangent composition solution
    void read_deltacompeq(const char * filename);                       // read composition difference from common tangent solution

    /* initialization */
    cl_int build_kernel(const char * kernel_file="kernel_float.cl");    // build OpenCL kernel
    void init_sim(const Type mean, const Type sigma);                   // initialize arrays and mem objects
    
    /* data transfer between host & device */
    cl_int write_mem();     // write Phi and Comp to buffer     (host   -> device)
    cl_int read_mem();      // read Phi and Comp from buffer    (device -> host)
    cl_int write_compa();   // write CompA to buffer            (host   -> device)
    cl_int write_compa(const size_t i_start, const size_t nstacks);
    cl_int write_deltacompeq(); // write DeltaCompEq to buffer
    
    /* simulation steps */
    void step(const Type dt); // do one step
    void steps(const Type dt, const unsigned int nsteps,
               const bool finish=true, const bool cputime=true); // do multiple steps
    
    void run(); // run simulation according to program

    void restart(const unsigned int t); // restart from #t steps; reset step counter to t
    
    /* write output file */
    void writefile();
    
};


#endif /* defined(__AllenCahnFD__simulator_2d__) */
