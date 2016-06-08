//
//  main.cpp
//  AllenCahnFD
//
//  Created by Yue Sun on 7/17/15.
//  Copyright (c) 2015 Yue Sun. All rights reserved.
//

#include <iostream>
#include <cstring>
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include "simulator_2d.hpp"
#include "simulator_3d.hpp"

int main(int argc, const char * argv[])
{
    // insert code here...
    std::cout << "Hello, World!\n";
    
    if (argc < 2)
    {
        std::cerr << "ERROR: No input file specified!" << std::endl;
        exit(EXIT_FAILURE);
    }
    
    int sim_dim;
    
    if (argc > 2)
        if (strcmp(argv[2],"-2")==0)
            sim_dim = 2;
    
    if (sim_dim == 2)
    {
        std::cout << "Starting 2D simulation ... \n";
        
        Simulator_2D<double> sim{};
        
        // Read input parameters
        sim.read_input(argv[1]);
        
        sim.init_cl(CL_DEVICE_TYPE_GPU, 1);
        
        sim.init_sim(0, 0.001);

        sim.read_init_cond("/Users/yue/Dropbox/Research/codes/phasefield/OpenCL/MoelansPhaseFieldFD_buffer/MoelansPhaseFieldFD/phia.bin", 
            "/Users/yue/Dropbox/Research/codes/phasefield/OpenCL/MoelansPhaseFieldFD_buffer/MoelansPhaseFieldFD/phib.bin",
            "/Users/yue/Dropbox/Research/codes/phasefield/OpenCL/MoelansPhaseFieldFD_buffer/MoelansPhaseFieldFD/comp.bin");
        sim.read_parabolic("/Users/yue/Dropbox/Research/codes/phasefield/fit_parabolic_f_matlab/para_coef.bin");
        sim.read_comp_phad("/Users/yue/Dropbox/Research/codes/phasefield/fit_parabolic_f_matlab/comp_phad.bin");
        
        sim.build_kernel("kernel_double_2d.cl");
        
        sim.run();
        
        return 0;
    }
    
    std::cout << "Starting 3D simulation ... \n";

    Simulator_3D<double> sim{};
    
    // Read input parameters
    sim.read_input(argv[1]);
    
    sim.init_cl(CL_DEVICE_TYPE_GPU, 1);
    
    sim.init_sim(0, 0.001);
    
    sim.read_init_cond("/Users/yue/Dropbox/Research/codes/phasefield/OpenCL/MoelansPhaseFieldFD_buffer/MoelansPhaseFieldFD/phia.bin",
                       "/Users/yue/Dropbox/Research/codes/phasefield/OpenCL/MoelansPhaseFieldFD_buffer/MoelansPhaseFieldFD/phib.bin",
                       "/Users/yue/Dropbox/Research/codes/phasefield/OpenCL/MoelansPhaseFieldFD_buffer/MoelansPhaseFieldFD/comp.bin");
    sim.read_parabolic("/Users/yue/Dropbox/Research/codes/phasefield/fit_parabolic_f_matlab/para_coef.bin");
    sim.read_comp_phad("/Users/yue/Dropbox/Research/codes/phasefield/fit_parabolic_f_matlab/comp_phad.bin");
    
    sim.build_kernel("kernel_double_3d.cl");
    
    sim.run();
    
    return 0;

}
