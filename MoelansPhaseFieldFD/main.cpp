//
//  main.cpp
//  AllenCahnFD
//
//  Created by Yue Sun on 7/17/15.
//  Copyright (c) 2015 Yue Sun. All rights reserved.
//

#include <iostream>
#include <string>
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
    
    std::string init_prefix, out_prefix;
//    init_prefix = "/Users/yue/Dropbox/Research/codes/phasefield/OpenCL/MoelansPhaseFieldFD_buffer/MoelansPhaseFieldFD/";
    init_prefix = "/Volumes/Ashwin_SSD_2014_03/phasefield_test/512/t_5_6/";
    out_prefix = "/Volumes/Ashwin_SSD_2014_03/phasefield_test/512/output/";
    
    if (sim_dim == 2)
    {
        std::cout << "Starting 2D simulation ... \n";
        
        Simulator_2D<double> sim{};
        
        // Read input parameters
        sim.read_input(argv[1]);
        
        sim.init_cl(CL_DEVICE_TYPE_GPU, 1);
        
        sim.init_sim(0, 0.001, out_prefix.c_str());

        sim.read_init_cond((init_prefix + "phia.bin").c_str(),
//                           (init_prefix + "phib.bin").c_str(),
                           (init_prefix + "comp.bin").c_str());
        sim.read_parabolic("/Users/yue/Dropbox/Research/codes/phasefield/preprocess/fit_parabolic_f_matlab/para_coef.bin");
        sim.read_comp_phad("/Users/yue/Dropbox/Research/codes/phasefield/preprocess/fit_parabolic_f_matlab/comp_phad.bin");
        
        sim.build_kernel("kernel_double_2d.cl");
        
        sim.run();
        
        return 0;
    }
    
    std::cout << "Starting 3D simulation ... \n";

    Simulator_3D<double> sim{};
    
    // Read input parameters
    sim.read_input(argv[1]);
    
    sim.init_cl(CL_DEVICE_TYPE_GPU, 1);
    
    sim.init_sim(0, 0.001, out_prefix.c_str());
    
    sim.read_init_cond((init_prefix + "phia.bin").c_str(),
//                           (init_prefix + "phib.bin").c_str(),
                       (init_prefix + "comp.bin").c_str());
    sim.read_parabolic("/Users/yue/Dropbox/Research/codes/phasefield/preprocess/fit_parabolic_f_matlab/para_coef.bin");
    sim.read_comp_phad("/Users/yue/Dropbox/Research/codes/phasefield/preprocess/fit_parabolic_f_matlab/comp_phad.bin");
    
    sim.build_kernel("kernel_double_3d.cl");
    
    sim.run();
    
    return 0;

}
