//
//  simulator_3d.cpp
//  AllenCahnFD
//
//  Created by Yue Sun on 7/20/15.
//  Copyright (c) 2015 Yue Sun. All rights reserved.
//

#include <cstdio>
#include <iostream>
#include <algorithm>

#include "simulator_3d.hpp"
#include "cl_common.hpp"
#include "input.hpp"
#include "randn.hpp"



template <typename Type>
Simulator_3D<Type>::Simulator_3D() : Simulator<Type>(0, 0, 0, 0) {}


template <typename Type>
Simulator_3D<Type>::Simulator_3D(const unsigned int & nx,
                              const unsigned int & ny,
                              const unsigned int & nz) : Simulator<Type>(nx, ny, nz, 2) {}


template <typename Type>
Simulator_3D<Type>::Simulator_3D(const unsigned int & nx,
                              const unsigned int & ny,
                              const unsigned int & nz,
                              const Type & a_2,
                              const Type & a_4,
                              const Type & M,
                              const Type & K,
                              const unsigned int & t_skip) : Simulator<Type>(nx, ny, nz, 2)
{
    _a_2 = a_2;
    _a_4 = a_4;
    _M = M;
    _K = K;
    _t_skip = t_skip;
}


template <typename Type>
Simulator_3D<Type>::~Simulator_3D()
{
    if (Simulator<Type>::_cl_initialized)
    {
        clReleaseKernel(_kernel_brac_3d);
        clReleaseKernel(_kernel_step_3d);
        clReleaseMemObject(_img_Phi);
        clReleaseMemObject(_img_Bracket);
        clReleaseMemObject(_img_PhiNext);
        clReleaseMemObject(_rotate_var);
    }
    
    delete [] Simulator<Type>::_data;
}



template <typename Type>
void Simulator_3D<Type>::read_input(const char * filename)
{
//    readfile(filename, Simulator<Type>::_dim.x, Simulator<Type>::_dim.y, Simulator<Type>::_dim.z,
//             _nt, _dx, _dt, _a_2, _a_4, _M, _K, _t_skip);
    Simulator<Type>::_size = Simulator<Type>::_dim.x * Simulator<Type>::_dim.y * Simulator<Type>::_dim.z;
}


template <typename Type>
cl_int Simulator_3D<Type>::build_kernel(const char * kernel_file)

{
    cl_int err;
    cl_program program;

    // Build program with source (filename) on device+context
    CHECK_RETURN_N(program, CreateProgram(Simulator<Type>::context, Simulator<Type>::device, kernel_file, err), err);
    CHECK_RETURN_N(_kernel_brac_3d, clCreateKernel(program, "brac_3d", &err), err);
    CHECK_RETURN_N(_kernel_step_3d, clCreateKernel(program, "step_3d", &err), err);

    //Create Images;
   
    cl_image_desc r_desc;
    r_desc.image_type=CL_MEM_OBJECT_IMAGE3D;
    r_desc.image_width=Simulator<Type>::_dim.x;
    r_desc.image_height=Simulator<Type>::_dim.y;
    r_desc.image_depth=Simulator<Type>::_dim.z;
    r_desc.image_row_pitch=0;
    r_desc.image_slice_pitch=0;
    r_desc.num_mip_levels=0;
    r_desc.num_samples=0;
    r_desc.buffer=NULL;
    
    cl_image_format fmt;
    fmt.image_channel_data_type=CL_FLOAT; /* DOES NOT WORK WITH DOUBLE */
    fmt.image_channel_order=CL_R;
    
    Type *v=(Type *)calloc(Simulator<Type>::_size,sizeof(Type));
    for(int i=0;i<Simulator<Type>::_size;i++)
    {
        v[i]=rand()%101/(Type)100;
    }
    CHECK_RETURN_N(_img_Phi,clCreateImage(Simulator<Type>::context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, &fmt, &r_desc, v, &err),err)
    CHECK_RETURN_N(_img_Bracket,clCreateImage(Simulator<Type>::context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, &fmt, &r_desc, v, &err),err)
    CHECK_RETURN_N(_img_PhiNext,clCreateImage(Simulator<Type>::context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, &fmt, &r_desc, v, &err),err)
    
    CHECK_ERROR(clSetKernelArg(_kernel_brac_3d, 0, sizeof(cl_mem), &_img_Phi))
    CHECK_ERROR(clSetKernelArg(_kernel_brac_3d, 1, sizeof(cl_mem), &_img_Bracket))
    CHECK_ERROR(clSetKernelArg(_kernel_brac_3d, 2, sizeof(Type), &_a_2))
    CHECK_ERROR(clSetKernelArg(_kernel_brac_3d, 3, sizeof(Type), &_a_4))
    CHECK_ERROR(clSetKernelArg(_kernel_brac_3d, 4, sizeof(Type), &_K))
    CHECK_ERROR(clSetKernelArg(_kernel_brac_3d, 5, sizeof(Type), &_dx))
    
    CHECK_ERROR(clSetKernelArg(_kernel_step_3d, 0, sizeof(cl_mem), &_img_Phi))
    CHECK_ERROR(clSetKernelArg(_kernel_step_3d, 1, sizeof(cl_mem), &_img_Bracket))
    CHECK_ERROR(clSetKernelArg(_kernel_step_3d, 2, sizeof(cl_mem), &_img_PhiNext))
    CHECK_ERROR(clSetKernelArg(_kernel_step_3d, 3, sizeof(Type), &_M))
    CHECK_ERROR(clSetKernelArg(_kernel_step_3d, 4, sizeof(Type), &_dx))
    CHECK_ERROR(clSetKernelArg(_kernel_step_3d, 5, sizeof(Type), &_dt))
    
    free(v);
    
    _local_size[0]=std::min((unsigned int)8,Simulator<Type>::_dim.x);
    _local_size[1]=std::min((unsigned int)8,Simulator<Type>::_dim.y);
    _local_size[2]=std::min((unsigned int)4,Simulator<Type>::_dim.z);
    
    _global_size[0]=Simulator<Type>::_dim.x;
    _global_size[1]=Simulator<Type>::_dim.y;
    _global_size[2]=Simulator<Type>::_dim.z;
    
    Simulator<Type>::_cl_initialized = true;
    
    return CL_SUCCESS;
}


template <typename Type>
cl_int Simulator_3D<Type>::write_mem()
{
    CHECK_ERROR(this->WriteArray(_img_Phi, Simulator<Type>::_data, _global_size[0], _global_size[1], _global_size[2]));
    return CL_SUCCESS;
}


template <typename Type>
cl_int Simulator_3D<Type>::read_mem()
{
    CHECK_ERROR(this->ReadArray(_img_Phi, Simulator<Type>::_data, _global_size[0], _global_size[1], _global_size[2]));
    return CL_SUCCESS;
}


template <typename Type>
void Simulator_3D<Type>::init_sim(const Type & mean, const Type & sigma)
{
    if (Simulator<Type>::_data == NULL)
        Simulator<Type>::_data = new Type[Simulator<Type>::_size];
    
    gauss(Simulator<Type>::_data, Simulator<Type>::_size, mean, sigma);
    
    write_mem();
}


template <typename Type>
void Simulator_3D<Type>::steps(const Type & dt, const unsigned int & nsteps, const bool finish, const bool cputime)
{
    if ((&dt != &_dt) && (dt != _dt))
    {
        _dt = dt;
        CHECK_ERROR_EXIT(clSetKernelArg(_kernel_step_3d, 5, sizeof(Type), &_dt))
    }
    
    // Variables for timing
    ttime_t t1=0, t2=0, t3=0;
    double s=0, s2=0;
    
    if(cputime)
        t1= getTime();
    
    for (unsigned int t=0; t<nsteps; t++)
    {
        // Calculate terms in the bracket
        CHECK_ERROR_EXIT(clEnqueueNDRangeKernel(Simulator<Type>::queue, _kernel_brac_3d, 3, NULL, _global_size, _local_size, 0, NULL, NULL));
        
        // Calculate laplacian of the bracket and do the time-stepping
        CHECK_ERROR_EXIT(clEnqueueNDRangeKernel(Simulator<Type>::queue, _kernel_step_3d, 3, NULL, _global_size, _local_size, 0, NULL, NULL));
        
        //ROTATE VARIABLES
        _rotate_var=_img_Phi;
        _img_Phi=_img_PhiNext;
        _img_PhiNext=_rotate_var;
        
        CHECK_ERROR_EXIT(clSetKernelArg(_kernel_brac_3d, 0, sizeof(cl_mem), &_img_Phi));
        CHECK_ERROR_EXIT(clSetKernelArg(_kernel_step_3d, 1, sizeof(cl_mem), &_img_PhiNext));
        
        
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
    
    //this->_disp_mem=img_Phi;
    Simulator<Type>::steps(_dt, nsteps);
    if(cputime)
        printf("It took %lfs to submit and %lfs to complete (%lfs per loops)\n",s,s2,s2/nsteps);
}


template <typename Type>
void Simulator_3D<Type>::run()
{
    if(Simulator<Type>::current_step==0)
        Simulator<Type>::writefile();
    
    ttime_t t0, t1;
    
    t0 = getTime();
    
    // for(unsigned int t=0; t < _nt; t+=_t_skip)
    for(unsigned int i=0; i < _nt/_t_skip; i++)
    {
        steps(_dt, _t_skip);
        read_mem();
        
        Simulator<Type>::writefile();
    }
    
    t1 = getTime();
    
    std::cout << "Total computation time: " << subtractTimes(t1, t0) << "s." << std::endl;
    
}



/* Explicit instantiation */
template class Simulator_3D<float>;
//template void Simulator_3D<float>::read_input(const char *);
//template void Simulator_3D<float>::init_sim(const float &, const float &);
//template Simulator_3D<float>::Simulator_3D();

template class Simulator_3D<double>;










