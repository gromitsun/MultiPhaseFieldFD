//
//  simulator_2d.cpp
//  AllenCahnFD
//
//  Created by Yue Sun on 7/20/15.
//  Copyright (c) 2015 Yue Sun. All rights reserved.
//

#include <cstdio>
#include <iostream>

#include "simulator_2d.hpp"
#include "cl_common.hpp"
#include "input.hpp"
#include "randn.hpp"
#include "util.hpp"


#define GET_T_IND(T) ((int)((T-_T_MIN)/_T_INC))

/******************** class initializers ********************/

template <typename Type>
Simulator_2D<Type>::Simulator_2D() : Simulator<Type>(0, 0, 0, 0) {}


template <typename Type>
Simulator_2D<Type>::Simulator_2D(const unsigned int nx,
                                 const unsigned int ny) : Simulator<Type>(nx, ny, 1, 2) {}


template <typename Type>
Simulator_2D<Type>::Simulator_2D(const unsigned int nx,
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
                                 const size_t T_NUM) : Simulator<Type>(nx, ny, 1, 2)
{
    /* Input parameters */
    // simulation parameters
    _dx = dx;
    _dt = dt;
    
    // physical parameters
    _Da = Da;
    _Db = Db;
    _sigma = sigma;
    _l = l;
    _T = T_start;
    _dT_dt = dT_dt;
    _t_skip = t_skip;

    // interpolation parameters
    _PHI_MIN = PHI_MIN;
    _COMP_MIN = COMP_MIN;
    _T_MIN = T_MIN;
    _PHI_INC = PHI_INC;
    _COMP_INC = COMP_INC;
    _T_INC = T_INC;
    _PHI_NUM = PHI_NUM;
    _COMP_NUM = COMP_NUM;
    _T_NUM = T_NUM;

    // phase field parameters
    _gamma = 1.5;    // = 1.5
    _kappa = 0.75*_sigma*_l;    // = 0.75*sigma*l
    _mk = 6*_sigma/_l;       // = 6*sigma/l
}


template <typename Type>
Simulator_2D<Type>::~Simulator_2D()
{
    if (Simulator<Type>::_cl_initialized)
    {
        clReleaseKernel(_kernel_step_2d);
        clReleaseMemObject(_mem_Phi);
        clReleaseMemObject(_mem_Comp);
        clReleaseMemObject(_mem_CompA);
        clReleaseMemObject(_mem_DeltaCompEq);
    }
    
    delete [] Simulator<Type>::_data;
}

/************************************************************/

/******************** loading data from file ********************/

template <typename Type>
void Simulator_2D<Type>::read_input(const char * filename)
{
    readfile(filename, Simulator<Type>::_dim.x, Simulator<Type>::_dim.y, Simulator<Type>::_dim.z,
             _nt, _dx, _dt, _Da, _Db, _sigma, _l, _T, _dT_dt, _t_skip,
             _PHI_MIN, _COMP_MIN, _T_MIN, _PHI_INC, _COMP_INC, _T_INC, _PHI_NUM, _COMP_NUM, _T_NUM);
    Simulator<Type>::_size = Simulator<Type>::_dim.x * Simulator<Type>::_dim.y;

    // phase field parameters
    _gamma = 1.5;    // = 1.5
    _kappa = 0.75*_sigma*_l;    // = 0.75*sigma*l
    _mk = 6*_sigma/_l;       // = 6*sigma/l
}


template <typename Type>
void Simulator_2D<Type>::read_init_cond(const char * filename_phi, const char * filename_comp)
{
    read_from_bin(filename_phi, _Phi, this->_size);
    read_from_bin(filename_comp, _Comp, this->_size);
}


template <typename Type>
void Simulator_2D<Type>::read_compa(const char * filename)
{
    read_from_bin(filename, _CompA, this->_size);
}


template <typename Type>
void Simulator_2D<Type>::read_deltacompeq(const char * filename)
{
    read_from_bin(filename, _DeltaCompEq, this->_size);
}

/************************************************************/

/******************** data transfer between host and device ********************/

template <typename Type>
cl_int Simulator_2D<Type>::write_mem()
{
    CHECK_ERROR(this->WriteArrayToBuffer(_mem_Phi, _Phi, _global_size[0], _global_size[1], _global_size[2]));
    CHECK_ERROR(this->WriteArrayToBuffer(_mem_Comp, _Comp, _global_size[0], _global_size[1], _global_size[2]));
    return CL_SUCCESS;
}


template <typename Type>
cl_int Simulator_2D<Type>::read_mem()
{
    CHECK_ERROR(this->ReadArrayFromBuffer(_mem_Phi, _Phi, _global_size[0], _global_size[1], _global_size[2]));
    CHECK_ERROR(this->ReadArrayFromBuffer(_mem_Comp, _Comp, _global_size[0], _global_size[1], _global_size[2]));
    return CL_SUCCESS;
}


template <typename Type>
cl_int Simulator_2D<Type>::write_compa()
{
    CHECK_ERROR(this->WriteArrayToBuffer(_mem_CompA, &_CompA[GET_T_IND(_T)], _PHI_NUM, _COMP_NUM, 2));
    return CL_SUCCESS;
}


template <typename Type>
cl_int Simulator_2D<Type>::write_compa(const size_t i_start, const size_t nstacks)
{
    CHECK_ERROR(this->WriteArrayToBuffer(_mem_CompA, &_CompA[i_start], _PHI_NUM, _COMP_NUM, nstacks));
    return CL_SUCCESS;
}

template <typename Type>
cl_int Simulator_2D<Type>::write_deltacompeq()
{
    CHECK_ERROR(this->WriteArrayToBuffer(_mem_DeltaCompEq, _DeltaCompEq, _T_NUM));
    return CL_SUCCESS;
}

/************************************************************/

/******************** initialize simulation ********************/

template <typename Type>
cl_int Simulator_2D<Type>::build_kernel(const char * kernel_file)

{
    cl_int err;
    cl_program program;

    // Build program with source (filename) on device+context
    CHECK_RETURN_N(program, CreateProgram(Simulator<Type>::context, Simulator<Type>::device, kernel_file, err), err);
    CHECK_RETURN_N(_kernel_step_2d, clCreateKernel(program, "step_2d", &err), err);

    //Create memory objects on device;
    
    CHECK_RETURN_N(_mem_Phi,clCreateBuffer(Simulator<Type>::context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, _global_size[0]*_global_size[1]*_global_size[2]*sizeof(Type), _Phi, &err),err)
    CHECK_RETURN_N(_mem_Comp,clCreateBuffer(Simulator<Type>::context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, _global_size[0]*_global_size[1]*_global_size[2]*sizeof(Type), _Comp, &err),err)
    
    CHECK_RETURN_N(_mem_CompA,clCreateBuffer(Simulator<Type>::context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, _PHI_NUM*_COMP_NUM*2*sizeof(Type), _CompA, &err),err)
    CHECK_RETURN_N(_mem_DeltaCompEq,clCreateBuffer(Simulator<Type>::context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, _T_NUM*sizeof(Type), _DeltaCompEq, &err),err)
    
    CHECK_ERROR(clSetKernelArg(_kernel_step_2d, 0, sizeof(cl_mem), &_mem_Phi))
    CHECK_ERROR(clSetKernelArg(_kernel_step_2d, 1, sizeof(cl_mem), &_mem_Comp))
    CHECK_ERROR(clSetKernelArg(_kernel_step_2d, 2, sizeof(cl_mem), &_mem_CompA))
    CHECK_ERROR(clSetKernelArg(_kernel_step_2d, 3, sizeof(cl_mem), &_mem_DeltaCompEq))
    CHECK_ERROR(clSetKernelArg(_kernel_step_2d, 4, sizeof(Type), &_T))
    CHECK_ERROR(clSetKernelArg(_kernel_step_2d, 5, sizeof(Type), &_mk))
    CHECK_ERROR(clSetKernelArg(_kernel_step_2d, 6, sizeof(Type), &_gamma))
    CHECK_ERROR(clSetKernelArg(_kernel_step_2d, 7, sizeof(Type), &_kappa))
    CHECK_ERROR(clSetKernelArg(_kernel_step_2d, 8, sizeof(Type), &_Da))
    CHECK_ERROR(clSetKernelArg(_kernel_step_2d, 9, sizeof(Type), &_Db))
    CHECK_ERROR(clSetKernelArg(_kernel_step_2d, 10, sizeof(Type), &_dx))
    CHECK_ERROR(clSetKernelArg(_kernel_step_2d, 11, sizeof(Type), &_dt))
    CHECK_ERROR(clSetKernelArg(_kernel_step_2d, 12, sizeof(Type), &_PHI_MIN))
    CHECK_ERROR(clSetKernelArg(_kernel_step_2d, 13, sizeof(Type), &_COMP_MIN))
    CHECK_ERROR(clSetKernelArg(_kernel_step_2d, 14, sizeof(Type), &_T_MIN))
    CHECK_ERROR(clSetKernelArg(_kernel_step_2d, 15, sizeof(Type), &_PHI_INC))
    CHECK_ERROR(clSetKernelArg(_kernel_step_2d, 16, sizeof(Type), &_COMP_INC))
    CHECK_ERROR(clSetKernelArg(_kernel_step_2d, 17, sizeof(Type), &_T_INC))
    CHECK_ERROR(clSetKernelArg(_kernel_step_2d, 18, sizeof(Type), &_PHI_NUM))
    CHECK_ERROR(clSetKernelArg(_kernel_step_2d, 19, sizeof(Type), &_COMP_NUM))
    
    Simulator<Type>::_cl_initialized = true;
    
    return CL_SUCCESS;
}

template <typename Type>
void Simulator_2D<Type>::init_sim(const Type mean, const Type sigma)
{
    /* allocate memory on host */
    _Phi = new Type[Simulator<Type>::_size];
    _Comp = new Type[Simulator<Type>::_size];
    _CompA = new Type[_PHI_NUM*_COMP_NUM*2];
    _DeltaCompEq = new Type[_T_NUM];
    
    /* initialize Phi & Comp with random numbers */
    gauss(_Phi, Simulator<Type>::_size, mean, sigma);
    gauss(_Comp, Simulator<Type>::_size, mean, sigma);
    
    /* set OpenCL calculation sizes */
    _local_size[0]=8;
    _local_size[1]=8;
    _local_size[2]=1;
    
    _global_size[0]=Simulator<Type>::_dim.x;
    _global_size[1]=Simulator<Type>::_dim.y;
    _global_size[2]=1;

    /* prepare directory for output */
    prep_dir("output/");
    
}

/************************************************************/

/******************** simulation steps ********************/

template <typename Type>
void Simulator_2D<Type>::step(const Type dt)
{
    steps(dt, 1, 0, 0);
}


template <typename Type>
void Simulator_2D<Type>::steps(const Type dt, const unsigned int nsteps, const bool finish, const bool cputime)
{
    if ((&dt != &_dt) && (dt != _dt))
    {
        _dt = dt;
        CHECK_ERROR_EXIT(clSetKernelArg(_kernel_step_2d, 11, sizeof(Type), &_dt))
    }
    
    // Variables for timing
    ttime_t t1=0, t2=0, t3=0;
    double s=0, s2=0;
    
    if(cputime)
        t1= getTime();
    
    
    Type dT = _dT_dt*_dt;
    Type T_target = _T + dT*nsteps;
    size_t i0 = GET_T_IND(_T);
    size_t i1 = GET_T_IND(T_target);
    
    unsigned int * t_list = new unsigned int [i1-i0+1];
    
    for (int i=0; i<i1-i0; i++)
    {
        t_list[i] = (unsigned int)(((i0+i+1)*_T_INC + _T_MIN - _T)/dT);
    }
    
    t_list[i1-i0] = nsteps;
    
    
    /* main loop */
    
    unsigned int t = 0;
    for (; t<t_list[0]; t++)
    {
        // Do calculation
        CHECK_ERROR_EXIT(clEnqueueNDRangeKernel(Simulator<Type>::queue, _kernel_step_2d, 2, NULL, _global_size, _local_size, 0, NULL, NULL));
    }
    
    for (int i=1; i<i1-i0+1; i++)
    {
        write_compa(i0+i, 2);
        for (; t<t_list[i]; t++)
        {
            // Do calculation
            CHECK_ERROR_EXIT(clEnqueueNDRangeKernel(Simulator<Type>::queue, _kernel_step_2d, 2, NULL, _global_size, _local_size, 0, NULL, NULL));
        }
    }
    
    for (int jj=123;jj<123+10;jj++)
        std::cout << _Phi[jj] <<"\t" << _Comp[jj] << std::endl;
    
    
    /* change temperature */
    _T = T_target;
    
    
    if(cputime)
    {
        t2=getTime();
        s = subtractTimes(t2,t1);
    }
    
    if(finish)
        clFinish(Simulator<Type>::queue);
    
    if(cputime)
    {
        t3=getTime();
        s2 = subtractTimes(t3,t1);
    }
    
    //this->_disp_mem=img_Phi;
    Simulator<Type>::steps(_dt, nsteps);
    if(cputime)
        printf("It took %lfs to submit and %lfs to complete (%lfs per loops)\n",s,s2,s2/nsteps);
}


template <typename Type>
void Simulator_2D<Type>::run()
{
    if(Simulator<Type>::current_step==0)
        writefile();
    
    ttime_t t0, t1;
    
    t0 = getTime();
    
    // for(unsigned int t=0; t < _nt; t+=_t_skip)
    for(unsigned int i=0; i < _nt/_t_skip; i++)
    {
        steps(_dt, _t_skip);
        read_mem();
        
        writefile();
    }
    
    t1 = getTime();
    
    std::cout << "Total computation time: " << subtractTimes(t1, t0) << "s." << std::endl;
    
}

template <typename Type>
void Simulator_2D<Type>::restart(const unsigned int t)
{
    _T += (t-Simulator<Type>::current_step)*_dT_dt*_dt;
    Simulator<Type>::restart(t, t*_dt);
}

/************************************************************/

/******************** output ********************/

template <typename Type>
void Simulator_2D<Type>::writefile()
{
    write2bin(time2fname("output/phi_", this->current_step), _Phi, this->_size);
    write2bin(time2fname("output/comp_", this->current_step), _Comp, this->_size);
}

/************************************************************/


/* Explicit instantiation */
template class Simulator_2D<float>;
template class Simulator_2D<double>;










