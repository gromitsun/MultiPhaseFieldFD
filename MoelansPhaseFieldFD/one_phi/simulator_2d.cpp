//
//  simulator_2d.cpp
//  AllenCahnFD
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


/******** functions for calculating model parameters ********/
inline double calc_g(double gamma)
{
    return std::sqrt(2.0 * (1.0 + 2.0*gamma)) / 6.0;
}


inline double calc_f0_max(double gamma)
{
    return 0.03125 + 0.0625 * gamma;
}


inline double calc_I(double gamma)
{
    return 1.0/std::sqrt(1.0 + 2.0*gamma);
}

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
        clReleaseKernel(_kernel_step_phi_2d);
        clReleaseKernel(_kernel_step_comp_2d);
        clReleaseMemObject(_mem_PhiA);
        clReleaseMemObject(_mem_Comp);
        clReleaseMemObject(_mem_PhiANext);
        clReleaseMemObject(_mem_U);
        clReleaseMemObject(_mem_M);
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
    double sqrt_f0_max = std::sqrt(calc_f0_max(_paras.gamma));
    double g = calc_g(_paras.gamma);
    _paras.kappa = sqrt_f0_max / g * _paras.sigma * _paras.l;
    _paras.mk = 1.0 / (sqrt_f0_max * g) * _paras.sigma / _paras.l;
    _paras.L0 = std::sqrt(2) * _paras.mk * g / (_paras.kappa*calc_I(_paras.gamma)); // L = L0 / zeta;
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
void Simulator_2D<Type>::read_parabolic(const char * filename)
{
    read_from_bin(filename, _para_coef, _paras.nT_data*6);
#ifdef DEBUG
    std::cout << "First 6 entries: \n";
    for (int i=0; i<6; i++)
        std::cout << _para_coef[i] << " ";
    std::cout << std::endl;
#endif
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
    CHECK_RETURN_N(program, CreateProgram(Simulator<Type>::context, Simulator<Type>::device, kernel_file, err), err);
    CHECK_RETURN_N(_kernel_step_phi_2d, clCreateKernel(program, "step_phi_2d", &err), err);
    CHECK_RETURN_N(_kernel_step_comp_2d, clCreateKernel(program, "step_comp_2d", &err), err);

    //Create memory objects on device;
    
    CHECK_RETURN_N(_mem_PhiA,clCreateBuffer(Simulator<Type>::context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, _global_size[0]*_global_size[1]*_global_size[2]*sizeof(Type), _PhiA, &err),err)
    CHECK_RETURN_N(_mem_Comp,clCreateBuffer(Simulator<Type>::context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, _global_size[0]*_global_size[1]*_global_size[2]*sizeof(Type), _Comp, &err),err)
    CHECK_RETURN_N(_mem_PhiANext,clCreateBuffer(Simulator<Type>::context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, _global_size[0]*_global_size[1]*_global_size[2]*sizeof(Type), _PhiA, &err),err)
    CHECK_RETURN_N(_mem_U,clCreateBuffer(Simulator<Type>::context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, _global_size[0]*_global_size[1]*_global_size[2]*sizeof(Type), _PhiA, &err),err)
    CHECK_RETURN_N(_mem_M,clCreateBuffer(Simulator<Type>::context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, _global_size[0]*_global_size[1]*_global_size[2]*sizeof(Type), _PhiA, &err),err)
    
    
    // Set kernel arguments
    CHECK_ERROR(clSetKernelArg(_kernel_step_phi_2d, 0, sizeof(cl_mem), &_mem_PhiA))
    CHECK_ERROR(clSetKernelArg(_kernel_step_phi_2d, 1, sizeof(cl_mem), &_mem_Comp))
    CHECK_ERROR(clSetKernelArg(_kernel_step_phi_2d, 2, sizeof(cl_mem), &_mem_PhiANext))
    CHECK_ERROR(clSetKernelArg(_kernel_step_phi_2d, 3, sizeof(cl_mem), &_mem_U))
    CHECK_ERROR(clSetKernelArg(_kernel_step_phi_2d, 4, sizeof(cl_mem), &_mem_M))
    CHECK_ERROR(clSetKernelArg(_kernel_step_phi_2d, 5, sizeof(Type), &_vars.a2))
    CHECK_ERROR(clSetKernelArg(_kernel_step_phi_2d, 6, sizeof(Type), &_vars.a1))
    CHECK_ERROR(clSetKernelArg(_kernel_step_phi_2d, 7, sizeof(Type), &_vars.a0))
    CHECK_ERROR(clSetKernelArg(_kernel_step_phi_2d, 8, sizeof(Type), &_vars.b2))
    CHECK_ERROR(clSetKernelArg(_kernel_step_phi_2d, 9, sizeof(Type), &_vars.b1))
    CHECK_ERROR(clSetKernelArg(_kernel_step_phi_2d, 10, sizeof(Type), &_vars.b0))
    CHECK_ERROR(clSetKernelArg(_kernel_step_phi_2d, 11, sizeof(Type), &_vars.L))
    CHECK_ERROR(clSetKernelArg(_kernel_step_phi_2d, 12, sizeof(Type), &_paras.Da))
    CHECK_ERROR(clSetKernelArg(_kernel_step_phi_2d, 13, sizeof(Type), &_paras.Db))
    CHECK_ERROR(clSetKernelArg(_kernel_step_phi_2d, 14, sizeof(Type), &_paras.mk))
    CHECK_ERROR(clSetKernelArg(_kernel_step_phi_2d, 15, sizeof(Type), &_paras.gamma))
    CHECK_ERROR(clSetKernelArg(_kernel_step_phi_2d, 16, sizeof(Type), &_paras.kappa))
    CHECK_ERROR(clSetKernelArg(_kernel_step_phi_2d, 17, sizeof(Type), &_paras.dx))
    CHECK_ERROR(clSetKernelArg(_kernel_step_phi_2d, 18, sizeof(Type), &_paras.dt))

    CHECK_ERROR(clSetKernelArg(_kernel_step_comp_2d, 0, sizeof(cl_mem), &_mem_Comp))
    CHECK_ERROR(clSetKernelArg(_kernel_step_comp_2d, 1, sizeof(cl_mem), &_mem_U))
    CHECK_ERROR(clSetKernelArg(_kernel_step_comp_2d, 2, sizeof(cl_mem), &_mem_M))
    CHECK_ERROR(clSetKernelArg(_kernel_step_comp_2d, 3, sizeof(Type), &_paras.dx))
    CHECK_ERROR(clSetKernelArg(_kernel_step_comp_2d, 4, sizeof(Type), &_paras.dt))

    Simulator<Type>::_cl_initialized = true;
    
    return CL_SUCCESS;
}

template <typename Type>
void Simulator_2D<Type>::init_sim(const Type mean, const Type sigma)
{
    /* allocate memory on host */
    _PhiA = new Type[Simulator<Type>::_size];
    _Comp = new Type[Simulator<Type>::_size];
    _para_coef = new Type[6 * _paras.nT_data];
    _comp_phad = new Type[2 * _paras.nT_data];
    
    /* initialize Phi & Comp with random numbers */
    gauss(_PhiA, Simulator<Type>::_size, mean, sigma);
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
    if ((&dt != &_paras.dt) && (dt != _paras.dt))
    {
        _paras.dt = dt;
        CHECK_ERROR_EXIT(clSetKernelArg(_kernel_step_phi_2d, 18, sizeof(Type), &_paras.dt))
        CHECK_ERROR_EXIT(clSetKernelArg(_kernel_step_comp_2d, 4, sizeof(Type), &_paras.dt))
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
        CHECK_ERROR_EXIT(clEnqueueNDRangeKernel(Simulator<Type>::queue, _kernel_step_phi_2d, 2, NULL, _global_size, _local_size, 0, NULL, NULL));
        CHECK_ERROR_EXIT(clEnqueueNDRangeKernel(Simulator<Type>::queue, _kernel_step_comp_2d, 2, NULL, _global_size, _local_size, 0, NULL, NULL));
    
        // rotate variables
        _rotate_var = _mem_PhiA;
        _mem_PhiA = _mem_PhiANext;
        _mem_PhiANext = _rotate_var;

        CHECK_ERROR_EXIT(clSetKernelArg(_kernel_step_phi_2d, 0, sizeof(cl_mem), &_mem_PhiA))
        CHECK_ERROR_EXIT(clSetKernelArg(_kernel_step_phi_2d, 2, sizeof(cl_mem), &_mem_PhiANext))

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
    /* change temperature */
    int T_idx = (int)((_vars.T - _paras.T_start_data)/_paras.dT_data);
    _vars.T_gibbs = T_idx * _paras.dT_data + _paras.T_start_data;
    _vars.T_gibbs_next = _vars.T_gibbs + ((_paras.dT_dt > 0) - (_paras.dT_dt < 0))*_paras.dT_recalc;
    
    /* recalculate equilibrium compositions */
    _vars.compa_eq = _comp_phad[2 * T_idx];
    _vars.compb_eq = _comp_phad[2 * T_idx + 1];
    _vars.delta_comp_eq = _vars.compb_eq - _vars.compa_eq;
    
    /* recalculate parabolic coefficients */
    _vars.a2 = _para_coef[6 * T_idx];
    _vars.a1 = _para_coef[6 * T_idx + 1];
    _vars.a0 = _para_coef[6 * T_idx + 2];
    _vars.b2 = _para_coef[6 * T_idx + 3];
    _vars.b1 = _para_coef[6 * T_idx + 4];
    _vars.b0 = _para_coef[6 * T_idx + 5];

    /* recalculate symmetric mobility coefficient M */
    _vars.m = 0.5 * (_paras.Da/_vars.a2 + _paras.Db/_vars.b2);
    
    /* recalculate kinetic coefficient L */
    _vars.L = _paras.L0 * _vars.m / (_vars.delta_comp_eq*_vars.delta_comp_eq);
    

    /* debug */
//   const double AS = 5.562064123036832e+09;
//   const double AL = 1.019435109224830e+10;
//   const double CS = -4.612094741994919e+09;
//   const double CL = -4.448563405669029e+09;
//   const double cS0 = 0.7821722753190940;
//   const double cL0 = 0.5704079007319450;
//   const double cSeq = 0.019862472879877200;
//   const double cLeq = 0.1544897158058190;
//   _vars.a2 = AS;
//   _vars.b2 = AL;
//   _vars.a1 = -cS0*AS;
//   _vars.b1 = -cL0*AL;
//   _vars.a0 = CS+0.5*AS*cS0*cS0;
//   _vars.b0 = CL+0.5*AL*cL0*cL0;
//   _vars.compa_eq = cSeq;
//   _vars.compb_eq = cLeq;
    /* debug */
    
    /* reset kernel arguments */
    CHECK_ERROR_EXIT(clSetKernelArg(_kernel_step_phi_2d, 5, sizeof(Type), &_vars.a2));
    CHECK_ERROR_EXIT(clSetKernelArg(_kernel_step_phi_2d, 6, sizeof(Type), &_vars.a1));
    CHECK_ERROR_EXIT(clSetKernelArg(_kernel_step_phi_2d, 7, sizeof(Type), &_vars.a0));
    CHECK_ERROR_EXIT(clSetKernelArg(_kernel_step_phi_2d, 8, sizeof(Type), &_vars.b2));
    CHECK_ERROR_EXIT(clSetKernelArg(_kernel_step_phi_2d, 9, sizeof(Type), &_vars.b1));
    CHECK_ERROR_EXIT(clSetKernelArg(_kernel_step_phi_2d, 10, sizeof(Type), &_vars.b0));
    CHECK_ERROR_EXIT(clSetKernelArg(_kernel_step_phi_2d, 11, sizeof(Type), &_vars.L));
    
#ifdef DEBUG
    std::cout << "Calculate Gibbs coefficients at T = " << _vars.T_gibbs << std::endl;
    std::cout << "a2 = " << _vars.a2 << " a1 = " << _vars.a1 << " a0 = " << _vars.a0 << std::endl;
    std::cout << "b2 = " << _vars.b2 << " b1 = " << _vars.b1 << " b0 = " << _vars.b0 << std::endl;
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
    double T = nt * _paras.dT_dt * _paras.dt;
    _vars.T_gibbs = (int)(T / _paras.dT_recalc) * _paras.dT_recalc;
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
    write2bin(time2fname("output/phia_", this->current_step), _PhiA, this->_size);
    write2bin(time2fname("output/comp_", this->current_step), _Comp, this->_size);
}

/************************************************************/


/* Explicit instantiation */
template class Simulator_2D<float>;
template class Simulator_2D<double>;










