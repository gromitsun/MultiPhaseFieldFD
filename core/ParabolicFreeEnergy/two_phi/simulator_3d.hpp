//
//  simulator_3d.h
//  AllenCahnFD
//
//  Created by Yue Sun on 7/20/15.
//  Copyright (c) 2015 Yue Sun. All rights reserved.
//

#ifndef __AllenCahnFD__simulator_3d__
#define __AllenCahnFD__simulator_3d__

#include "simulator.hpp"
#include "parameter_type.hpp"


template <typename Type>
class Simulator_3D : public Simulator<Type>
{
private:
    /* Input parameters */
    Parameter<Type> _paras;
    
    /* Simulation variables */
    Variable<Type> _vars;
    
    /* Output directory */
    std::string _outdir = "output/";
    
    /* Host arrays */
    Type * _PhiA;
    Type * _PhiB;
    Type * _Comp;
    Type * _para_coef;
    Type * _comp_phad;
    
    
    /* OpenCL variables */
    
    // OpenCL memory objects
    cl_mem _mem_PhiA;
    cl_mem _mem_PhiB;
    cl_mem _mem_Comp;
    cl_mem _mem_PhiANext;
    cl_mem _mem_PhiBNext;
    cl_mem _rotate_var;
    cl_mem _mem_U;
    cl_mem _mem_M;
    
    // OpenCL kernel
    cl_kernel _kernel_step_phi_3d;
    cl_kernel _kernel_step_comp_3d;
    
    // OpenCL sizes
    size_t _local_size[3];
    size_t _global_size[3];
    
public:
    Simulator_3D();
    Simulator_3D(const unsigned int & nx,
                 const unsigned int & ny,
                 const unsigned int & nz);
    Simulator_3D(const unsigned int & nx,
                 const unsigned int & ny,
                 const unsigned int & nz,
                 const Parameter<Type> & paras);
    ~Simulator_3D();
    
    /* read input from files */
    void read_input(const char * filename);                             // read input parameters
    void read_init_cond(const char * phia_file, const char * phib_file, const char * comp_file); // read initial condition from data
    void read_parabolic(const char * filename); // read parabolic free energy coefficients
    void read_comp_phad(const char * filename); // read equilibrium compositions
    
    /* calculate derived model parameters */
    void calc_paras();
    
    /* initialization */
    cl_int build_kernel(const char * kernel_file="kernel_float.cl");    // build OpenCL kernel
    void init_sim(const Type mean, const Type sigma, const char * outdir="output/");                   // initialize arrays and mem objects
    
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
    
    void restart(const unsigned int nt); // restart from #t steps; reset step counter to t
    
    /* write output file */
    void writefile();
};


#endif /* defined(__AllenCahnFD__simulator_3d__) */
