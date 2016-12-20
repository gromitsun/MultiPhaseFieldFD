//
//  simulator_2d.cpp
//  SteinbachPhaseFieldFD
//
//  Created by Yue Sun on 7/20/15.
//  Copyright (c) 2015 Yue Sun. All rights reserved.
//

#include <cstdio>
#include <iostream>
#include <cmath>

#include "simulator_2d.hpp"
#include "cl_common.hpp"
#include "input.hpp"
#include "randn.hpp"
#include "util.hpp"
#include "alcu_gibbs.hpp"

#define LINEAR_TEMP_INTERP
#define ONE_D_SIM

/************************************************************/

/******************** class initializers ********************/

template <typename Type>
Simulator_2D<Type>::Simulator_2D() : Simulator<Type>(0, 0, 0, 0) {}


template <typename Type>
Simulator_2D<Type>::Simulator_2D(const unsigned int nx,
                                 const unsigned int ny) : Simulator<Type>(nx, ny, 1, 2) {}


template <typename Type>
Simulator_2D<Type>::Simulator_2D(const unsigned int nx,
                                 const unsigned int ny,
                                 const Parameter<Type> & paras) : Simulator<Type>(nx, ny, 1, 2)
{
    /* Input parameters */
    _paras = paras;

    // phase field parameters    
    calc_paras();
}


template <typename Type>
Simulator_2D<Type>::~Simulator_2D()
{
    if (Simulator<Type>::_cl_initialized)
    {
        clReleaseKernel(_kernel_step_2d);
        clReleaseMemObject(_mem_PhiA);
        clReleaseMemObject(_mem_Comp);
        clReleaseMemObject(_mem_PhiANext);
        clReleaseMemObject(_mem_CompNext);
    }
    
    delete [] _PhiA;
    delete [] _Comp;
    delete [] _para_coef;
    delete [] _comp_phad;
}

/************************************************************/


/************ calculate derived model parameters ************/

template <typename Type>
void Simulator_2D<Type>::calc_paras()
{
    _paras.kappa = 4.0 / (M_PI * M_PI) * _paras.sigma * _paras.l;
    _paras.mk = 4.0 * _paras.sigma / _paras.l;
    double a2 = 0.25 * (std::sqrt(2)-1) * M_PI;
    _paras.L0 = _paras.mk / (6.0 * _paras.kappa * a2); // L = L0 / zeta;
}

/************************************************************/


/******************** loading data from file ********************/

template <typename Type>
void Simulator_2D<Type>::read_input(const char * filename)
{
    readfile(filename, Simulator<Type>::_dim.x, Simulator<Type>::_dim.y, 
             Simulator<Type>::_dim.z, _paras);
    Simulator<Type>::_size = Simulator<Type>::_dim.x * Simulator<Type>::_dim.y;

    // phase field parameters
    calc_paras();
}


template <typename Type>
void Simulator_2D<Type>::read_init_cond(const char * filename_phia, const char * filename_comp)
{
    read_from_bin(filename_phia, _PhiA, this->_size);
    read_from_bin(filename_comp, _Comp, this->_size);
}


template <typename Type>
void Simulator_2D<Type>::read_comp_phad(const char * filename)
{
    read_from_bin(filename, _comp_phad, _paras.nT_data*2);
#ifdef DEBUG
    std::cout << "First 6 entries: \n";
    for (int i=0; i<6; i++)
        std::cout << _comp_phad[i] << " ";
    std::cout << std::endl;
#endif
}


/************************************************************/

/******************** data transfer between host and device ********************/

template <typename Type>
cl_int Simulator_2D<Type>::write_mem()
{
    CHECK_ERROR(this->WriteArrayToBuffer(_mem_PhiA, _PhiA, _global_size[0], _global_size[1], _global_size[2]));
    CHECK_ERROR(this->WriteArrayToBuffer(_mem_Comp, _Comp, _global_size[0], _global_size[1], _global_size[2]));
    return CL_SUCCESS;
}


template <typename Type>
cl_int Simulator_2D<Type>::read_mem()
{
    CHECK_ERROR(this->ReadArrayFromBuffer(_mem_PhiA, _PhiA, _global_size[0], _global_size[1], _global_size[2]));
    CHECK_ERROR(this->ReadArrayFromBuffer(_mem_Comp, _Comp, _global_size[0], _global_size[1], _global_size[2]));
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
    CHECK_RETURN(program, CreateProgram(Simulator<Type>::context, Simulator<Type>::device, kernel_file, err), err);
    CHECK_RETURN(_kernel_step_2d, clCreateKernel(program, "step_2d", &err), err);

    //Create memory objects on device;
    
    CHECK_RETURN(_mem_PhiA,clCreateBuffer(Simulator<Type>::context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, _global_size[0]*_global_size[1]*_global_size[2]*sizeof(Type), _PhiA, &err),err)
    CHECK_RETURN(_mem_Comp,clCreateBuffer(Simulator<Type>::context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, _global_size[0]*_global_size[1]*_global_size[2]*sizeof(Type), _Comp, &err),err)
    CHECK_RETURN(_mem_PhiANext,clCreateBuffer(Simulator<Type>::context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, _global_size[0]*_global_size[1]*_global_size[2]*sizeof(Type), _PhiA, &err),err)
    CHECK_RETURN(_mem_CompNext,clCreateBuffer(Simulator<Type>::context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, _global_size[0]*_global_size[1]*_global_size[2]*sizeof(Type), _PhiA, &err),err)
    
    
    // Set kernel arguments
    unsigned long int local_ngrids = (_local_size[0]+2)*(_local_size[1]+2)*(_local_size[2]+2);
    
    // input/output arrays
    CHECK_ERROR(clSetKernelArg(_kernel_step_2d,  0, sizeof(cl_mem), &_mem_PhiA))
    CHECK_ERROR(clSetKernelArg(_kernel_step_2d,  1, sizeof(cl_mem), &_mem_Comp))
    CHECK_ERROR(clSetKernelArg(_kernel_step_2d,  2, sizeof(cl_mem), &_mem_PhiANext))
    CHECK_ERROR(clSetKernelArg(_kernel_step_2d,  3, sizeof(cl_mem), &_mem_CompNext))
    CHECK_ERROR(clSetKernelArg(_kernel_step_2d,  4, sizeof(Type)*local_ngrids, NULL))
    CHECK_ERROR(clSetKernelArg(_kernel_step_2d,  5, sizeof(Type)*local_ngrids, NULL))
    // coefficients of the free energy functions
    CHECK_ERROR(clSetKernelArg(_kernel_step_2d,  6, sizeof(Type), &_vars.a[0]))
    CHECK_ERROR(clSetKernelArg(_kernel_step_2d,  7, sizeof(Type), &_vars.a[1]))
    CHECK_ERROR(clSetKernelArg(_kernel_step_2d,  8, sizeof(Type), &_vars.a[2]))
    CHECK_ERROR(clSetKernelArg(_kernel_step_2d,  9, sizeof(Type), &_vars.a[3]))
    CHECK_ERROR(clSetKernelArg(_kernel_step_2d, 10, sizeof(Type), &_vars.a[4]))
    CHECK_ERROR(clSetKernelArg(_kernel_step_2d, 11, sizeof(Type), &_vars.b[0]))
    CHECK_ERROR(clSetKernelArg(_kernel_step_2d, 12, sizeof(Type), &_vars.b[1]))
    CHECK_ERROR(clSetKernelArg(_kernel_step_2d, 13, sizeof(Type), &_vars.b[2]))
    CHECK_ERROR(clSetKernelArg(_kernel_step_2d, 14, sizeof(Type), &_vars.b[3]))
    CHECK_ERROR(clSetKernelArg(_kernel_step_2d, 15, sizeof(Type), &_vars.b[4]))
    CHECK_ERROR(clSetKernelArg(_kernel_step_2d, 16, sizeof(Type), &_vars.b[5]))
    // R * T
    CHECK_ERROR(clSetKernelArg(_kernel_step_2d, 17, sizeof(Type), &_vars.RT))
    // equilibrium compositions
    CHECK_ERROR(clSetKernelArg(_kernel_step_2d, 18, sizeof(Type), &_vars.compa_eq))
    CHECK_ERROR(clSetKernelArg(_kernel_step_2d, 19, sizeof(Type), &_vars.compb_eq))
    // partition coefficient (f2a/f2b) at equilibrium compositions (c.f. Steinbach 2006 PRE)
    CHECK_ERROR(clSetKernelArg(_kernel_step_2d, 20, sizeof(Type), &_vars.kba))
    // diffusion coefficients
    CHECK_ERROR(clSetKernelArg(_kernel_step_2d, 21, sizeof(Type), &_paras.Da))
    CHECK_ERROR(clSetKernelArg(_kernel_step_2d, 22, sizeof(Type), &_paras.Db))
    // phase field parameters
    CHECK_ERROR(clSetKernelArg(_kernel_step_2d, 23, sizeof(Type), &_vars.L))
    CHECK_ERROR(clSetKernelArg(_kernel_step_2d, 24, sizeof(Type), &_paras.mk))
    CHECK_ERROR(clSetKernelArg(_kernel_step_2d, 25, sizeof(Type), &_paras.kappa))
    // simulation parameters
    CHECK_ERROR(clSetKernelArg(_kernel_step_2d, 26, sizeof(Type), &_paras.dx))
    CHECK_ERROR(clSetKernelArg(_kernel_step_2d, 27, sizeof(Type), &_paras.dt))

    Simulator<Type>::_cl_initialized = true;
    
    return CL_SUCCESS;
}

template <typename Type>
void Simulator_2D<Type>::init_sim(const Type mean, const Type sigma, const char * outdir)
{
    /* allocate memory on host */
    _PhiA = new Type[Simulator<Type>::_size];
    _Comp = new Type[Simulator<Type>::_size];
    _comp_phad = new Type[2 * _paras.nT_data];
    
    /* initialize Phi & Comp with random numbers */
    gauss(_PhiA, Simulator<Type>::_size, mean, sigma);
    gauss(_Comp, Simulator<Type>::_size, mean, sigma);
    
    /* set OpenCL calculation sizes */
#ifndef ONE_D_SIM // 2D local size
    _local_size[0]=std::min((unsigned int)8,Simulator<Type>::_dim.x);
    _local_size[1]=std::min((unsigned int)8,Simulator<Type>::_dim.y);
    _local_size[2]=std::min((unsigned int)1,Simulator<Type>::_dim.z);
#else // 1D local size
    _local_size[0]=std::min((unsigned int)128,Simulator<Type>::_dim.x);
    _local_size[1]=std::min((unsigned int)1,Simulator<Type>::_dim.y);
    _local_size[2]=std::min((unsigned int)1,Simulator<Type>::_dim.z);
#endif
    
    _global_size[0]=Simulator<Type>::_dim.x;
    _global_size[1]=Simulator<Type>::_dim.y;
    _global_size[2]=1;

    /* prepare directory for output */
    _outdir = outdir;
    prep_dir(outdir);
    
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
    if ((&dt != &_paras.dt) && (dt != _paras.dt))
    {
        _paras.dt = dt;
        CHECK_ERROR_EXIT(clSetKernelArg(_kernel_step_2d, 27, sizeof(Type), &_paras.dt))
    }
    
    // Variables for timing
    ttime_t t1=0, t2=0, t3=0;
    double s=0, s2=0;
    
    if(cputime)
        t1= getTime();
    
    
    Type dT = _paras.dT_dt*_paras.dt;
    
    /* main loop */
    
    for (unsigned int t = 0; t<nsteps; t++)
    {
        // Do calculation
        CHECK_ERROR_EXIT(clEnqueueNDRangeKernel(Simulator<Type>::queue, _kernel_step_2d, 2, NULL, _global_size, _local_size, 0, NULL, NULL));
    
        // rotate variables
        _rotate_var = _mem_PhiA;
        _mem_PhiA = _mem_PhiANext;
        _mem_PhiANext = _rotate_var;

        _rotate_var = _mem_Comp;
        _mem_Comp = _mem_CompNext;
        _mem_CompNext = _rotate_var;
        
        CHECK_ERROR_EXIT(clSetKernelArg(_kernel_step_2d, 0, sizeof(cl_mem), &_mem_PhiA))
        CHECK_ERROR_EXIT(clSetKernelArg(_kernel_step_2d, 2, sizeof(cl_mem), &_mem_PhiANext))
        CHECK_ERROR_EXIT(clSetKernelArg(_kernel_step_2d, 1, sizeof(cl_mem), &_mem_Comp))
        CHECK_ERROR_EXIT(clSetKernelArg(_kernel_step_2d, 3, sizeof(cl_mem), &_mem_CompNext))

        // increase global counter
        Simulator<Type>::steps(_paras.dt, 1);
        
        /* change temperature */
        _vars.T += dT;

        if (dT && ((dT > 0) ? (_vars.T >= _vars.T_gibbs_next) : (_vars.T <= _vars.T_gibbs_next)))
        {
#ifdef DEBUG
            std::cout << "Reached T recalc limit: T_actual = " << _vars.T
                      << " T_gibbs = " << _vars.T_gibbs
                      << " T_next = " << _vars.T_gibbs_next << std::endl;
#endif
            // recalculate parabolic coefficients in free energy functions
            set_temp();
            
        }


    }
    
    
    
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
    
    
    if(cputime)
        printf("It took %lfs to submit and %lfs to complete (%lfs per loops)\n",s,s2,s2/nsteps);
}


template <typename Type>
void Simulator_2D<Type>::set_temp(const Type T)
{
    /* change temperature */
//    if (T != _vars.T)
//    {
        _vars.T = T;
        set_temp();
//    }
    
}

template <typename Type>
void Simulator_2D<Type>::set_temp()
{
//    if (_vars.T != _vars.T_gibbs)
//    {
    
#ifndef LINEAR_TEMP_INTERP
    /* change temperature */
    int T_idx = (int)std::round((_vars.T - _paras.T_start_data)/_paras.dT_data);
    _vars.T_gibbs = T_idx * _paras.dT_data + _paras.T_start_data;
    _vars.T_gibbs_next = _vars.T_gibbs + ((_paras.dT_dt > 0) - (_paras.dT_dt < 0))*_paras.dT_recalc;
    
    /* recalculate equilibrium compositions */
    _vars.compa_eq = _comp_phad[2 * T_idx];
    _vars.compb_eq = _comp_phad[2 * T_idx + 1];
    _vars.delta_comp_eq = _vars.compb_eq - _vars.compa_eq;
#else
    /* change temperature */
    _vars.T_gibbs = _vars.T;
    _vars.T_gibbs_next = _vars.T_gibbs + ((_paras.dT_dt > 0) - (_paras.dT_dt < 0))*_paras.dT_recalc;
    
    /* calculate linear interpolation coefficients */
    int T_idx = (int)((_vars.T - _paras.T_start_data)/_paras.dT_data);
    double T_data = T_idx * _paras.dT_data + _paras.T_start_data;
    double h2 = (_vars.T - T_data) / _paras.dT_data;
    double h1 = 1 - h2;
    
    /* recalculate equilibrium compositions */
    _vars.compa_eq = h1 * _comp_phad[2 * T_idx] + h2 * _comp_phad[2 * T_idx + 2];
    _vars.compb_eq = h1 * _comp_phad[2 * T_idx + 1] + h2 * _comp_phad[2 * T_idx + 3];;
    _vars.delta_comp_eq = _vars.compb_eq - _vars.compa_eq;
#endif
    
    /* recalculate R * T */
    _vars.RT = R * _vars.T_gibbs / Vma; // normalize coefficients by Vm
    
    /* recalculate free energy function coefficients */
    calc_coef(_vars.a, _vars.b, _vars.T_gibbs);
    for (int i=0; i<5; i++){
        _vars.a[i] /= Vma; // normalize coefficients by Vm
    }
    for (int i=0; i<6; i++){
        _vars.b[i] /= Vmb; // normalize coefficients by Vm
    }
    
    /* calculate 2nd derivatives at equilibrium compositions */
    _vars.f2a = _vars.RT / (_vars.compa_eq*(1-_vars.compa_eq))
    + 2*_vars.a[2]
    + 6*_vars.a[3] * _vars.compa_eq
    + 12*_vars.a[4] * _vars.compa_eq*_vars.compa_eq;
    _vars.f2b = _vars.RT / (_vars.compb_eq*(1-_vars.compb_eq))
    + 2*_vars.b[2]
    + 6*_vars.b[3] * _vars.compb_eq
    + 12*_vars.b[4] * _vars.compb_eq*_vars.compb_eq
    + 20*_vars.b[5] * _vars.compb_eq*_vars.compb_eq*_vars.compb_eq;
    _vars.kba = _vars.f2a / _vars.f2b;
    
    /* recalculate symmetric mobility coefficient M */
    _vars.m = 0.5 * (_paras.Da/_vars.f2a + _paras.Db/_vars.f2b);
    
    /* recalculate kinetic coefficient L */
    _vars.L = _paras.L0 * _vars.m / (_vars.delta_comp_eq*_vars.delta_comp_eq);
    
    /* reset kernel arguments */
    // coefficients of the free energy functions
    CHECK_ERROR_EXIT(clSetKernelArg(_kernel_step_2d,  6, sizeof(Type), &_vars.a[0]));
    CHECK_ERROR_EXIT(clSetKernelArg(_kernel_step_2d,  7, sizeof(Type), &_vars.a[1]));
    CHECK_ERROR_EXIT(clSetKernelArg(_kernel_step_2d,  8, sizeof(Type), &_vars.a[2]));
    CHECK_ERROR_EXIT(clSetKernelArg(_kernel_step_2d,  9, sizeof(Type), &_vars.a[3]));
    CHECK_ERROR_EXIT(clSetKernelArg(_kernel_step_2d, 10, sizeof(Type), &_vars.a[4]));
    CHECK_ERROR_EXIT(clSetKernelArg(_kernel_step_2d, 11, sizeof(Type), &_vars.b[0]));
    CHECK_ERROR_EXIT(clSetKernelArg(_kernel_step_2d, 12, sizeof(Type), &_vars.b[1]));
    CHECK_ERROR_EXIT(clSetKernelArg(_kernel_step_2d, 13, sizeof(Type), &_vars.b[2]));
    CHECK_ERROR_EXIT(clSetKernelArg(_kernel_step_2d, 14, sizeof(Type), &_vars.b[3]));
    CHECK_ERROR_EXIT(clSetKernelArg(_kernel_step_2d, 15, sizeof(Type), &_vars.b[4]));
    CHECK_ERROR_EXIT(clSetKernelArg(_kernel_step_2d, 16, sizeof(Type), &_vars.b[5]));
    // R * T
    CHECK_ERROR_EXIT(clSetKernelArg(_kernel_step_2d, 17, sizeof(Type), &_vars.RT));
    // equilibrium compositions
    CHECK_ERROR_EXIT(clSetKernelArg(_kernel_step_2d, 18, sizeof(Type), &_vars.compa_eq));
    CHECK_ERROR_EXIT(clSetKernelArg(_kernel_step_2d, 19, sizeof(Type), &_vars.compb_eq));
    // partition coefficient (f2a/f2b) at equilibrium compositions (c.f. Steinbach 2006 PRE)
    CHECK_ERROR_EXIT(clSetKernelArg(_kernel_step_2d, 20, sizeof(Type), &_vars.kba));
    // phase field parameters
    CHECK_ERROR_EXIT(clSetKernelArg(_kernel_step_2d, 23, sizeof(Type), &_vars.L));
    
#ifdef DEBUG
    std::cout << "Calculate Gibbs coefficients at T = " << _vars.T_gibbs << std::endl;
    std::cout << "a2 = " << _vars.a[2] << " a1 = " << _vars.a[1] << " a0 = " << _vars.a[0] << std::endl;
    std::cout << "b2 = " << _vars.b[2] << " b1 = " << _vars.b[1]<< " b0 = " << _vars.b[0] << std::endl;
    std::cout << "compa_eq = " << _vars.compa_eq << " compb_eq = " << _vars.compb_eq << std::endl;
    std::cout << "Next recalculation at T = " << _vars.T_gibbs_next << std::endl;
#endif
//    }
    
}


template <typename Type>
void Simulator_2D<Type>::run()
{
    if(Simulator<Type>::current_step==0)
    {
#ifdef DEBUG
        for (int i=0;i<10;i++)
        {
            std::cout << _PhiA[i+this->_dim.x/2-5+(this->_dim.y/2)*this->_dim.x] << "\t";
            std::cout << _Comp[i+this->_dim.x/2-5+(this->_dim.y/2)*this->_dim.x] << "\n";
        }
#endif
        writefile();
        
        /* set temperature to T_start */
        set_temp(_paras.T_start);
    }
    
    ttime_t t0, t1;
    
    t0 = getTime();
    
    // for(unsigned int t=0; t < _nt; t+=_t_skip)
    for(unsigned int i=0; i < _paras.nt/_paras.t_skip; i++)
    {
        steps(_paras.dt, _paras.t_skip);
        read_mem();
        
#ifdef DEBUG
        for (int i=0;i<10;i++)
        {
            std::cout << _PhiA[i+this->_dim.x/2-5+(this->_dim.y/2)*this->_dim.x] << "\t";
            std::cout << _Comp[i+this->_dim.x/2-5+(this->_dim.y/2)*this->_dim.x] << "\n";
        }
#endif
        
        writefile();
    }
    
    t1 = getTime();
    
    std::cout << "Total computation time: " << subtractTimes(t1, t0) << "s." << std::endl;
    
}

template <typename Type>
void Simulator_2D<Type>::restart(const unsigned int nt)
{
    double t = nt * _paras.dt;
    double T = _paras.T_start + nt * _paras.dT_dt * _paras.dt;
    _vars.T_gibbs = ((int)(T / _paras.dT_recalc) + (_paras.dT_dt < 0))* _paras.dT_recalc;
    std::cout << "Restarting from step = " << nt 
              << ", t = " << t
              << ", T = " << T
              << ", T_gibbs = " << _vars.T_gibbs << std::endl;
    set_temp(_vars.T_gibbs);
    _vars.T = T;
    Simulator<Type>::restart(nt, t);
}

/************************************************************/

/******************** output ********************/

template <typename Type>
void Simulator_2D<Type>::writefile()
{
    write2bin(time2fname((_outdir + "phia_").c_str(), this->current_step), _PhiA, this->_size);
    write2bin(time2fname((_outdir + "comp_").c_str(), this->current_step), _Comp, this->_size);
}

/************************************************************/


/* Explicit instantiation */
template class Simulator_2D<float>;
template class Simulator_2D<double>;










