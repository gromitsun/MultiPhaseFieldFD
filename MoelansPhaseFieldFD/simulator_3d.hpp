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

template <typename Type>
class Simulator_3D : public Simulator<Type>
{
private:
    // physical parameters
    Type _a_2;
    Type _a_4;
    Type _M;
    Type _K;
    // simulation parameters
    Type _dx;
    Type _dt;
    unsigned int _nt;
    unsigned int _t_skip;
    
    cl_mem _img_Phi;
    cl_mem _img_Bracket;
    cl_mem _img_PhiNext;
    
    cl_mem _rotate_var;
    
    cl_kernel _kernel_brac_3d;
    cl_kernel _kernel_step_3d;
    
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
                 const Type & a_2,
                 const Type & a_4,
                 const Type & M,
                 const Type & K,
                 const unsigned int & t_skip);
    ~Simulator_3D();
    
    void read_input(const char * filename);
    
    cl_int build_kernel(const char * kernel_file="kernel_float_3d.cl");
    void init_sim(const Type & mean, const Type & sigma);
    cl_int write_mem();
    cl_int read_mem();
    
    void step(const Type & dt);
    void steps(const Type & dt, const unsigned int & nsteps, const bool finish=true, const bool cputime=true);
    
    void run();
};


#endif /* defined(__AllenCahnFD__simulator_3d__) */
