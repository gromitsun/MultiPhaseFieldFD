//
//  main.cpp
//  SteinbachPhaseFieldFD
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
    
    std::string init_prefix, out_prefix, input_prefix;
    init_prefix = "/Volumes/Ashwin_SSD_2014_03/phasefield_test_2/flat/sharp_alpha_0.2/";
    input_prefix = "/Volumes/Ashwin_SSD_2014_03/phasefield_test_2/";
    sim_dim = 3;
    
//    init_prefix = "/Volumes/Ashwin_SSD_2014_03/phasefield_test/gridrec/256/t_27_smooth/";
//    input_prefix = "/Volumes/Ashwin_SSD_2014_03/phasefield_test/gridrec/256/2/";
    
    out_prefix = input_prefix + "output/";
    
    if (sim_dim == 2)
    {
        std::cout << "Starting 2D simulation ... \n";
        
        Simulator_2D<double> sim{};
        
        // Read input parameters
        sim.read_input((input_prefix + argv[1]).c_str());
        
        // Initialize program
        sim.init_cl(CL_DEVICE_TYPE_GPU, 1);
        
        sim.init_sim(0, 0.001, out_prefix.c_str());
        
        sim.build_kernel("kernel_double_2d.cl"); // contains memcopy host -> device
        
        // Read data from file
        sim.read_init_cond((init_prefix + "phia.bin").c_str(),
//                           (init_prefix + "phib.bin").c_str(),
                           (init_prefix + "comp.bin").c_str());
        sim.write_mem();
        sim.read_parabolic("/Users/yue/Dropbox/Research/codes/phasefield/Moelans_phase_field/preprocess/fit_parabolic_f_matlab/output/T_817_907/para_coef.bin");
        sim.read_comp_phad("/Users/yue/Dropbox/Research/codes/phasefield/Moelans_phase_field/preprocess/fit_parabolic_f_matlab/output/T_817_907/comp_phad.bin");
        
        // Restart from previous calculations
//        sim.read_init_cond((out_prefix + "phia_1000000.bin").c_str(), (out_prefix + "comp_1000000.bin").c_str());
//        sim.write_mem();
//        sim.restart(1000000); // set step counter
        
        // Start simulation steps
        sim.run();
        
        return 0;
    }
    
    std::cout << "Starting 3D simulation ... \n";

    Simulator_3D<double> sim{};
    
    // Read input parameters
    sim.read_input((input_prefix + argv[1]).c_str());
    
    // Initialize program
    sim.init_cl(CL_DEVICE_TYPE_GPU, 1);
    
    sim.init_sim(0, 0.001, out_prefix.c_str());
    
    sim.build_kernel("kernel_double_3d.cl"); // contains memcopy host -> device
    
    // Read data from file
    sim.read_init_cond((init_prefix + "phia.bin").c_str(),
//                           (init_prefix + "phib.bin").c_str(),
                       (init_prefix + "comp.bin").c_str());
    sim.write_mem();
    sim.read_parabolic("/Users/yue/Dropbox/Research/codes/phasefield/Moelans_phase_field/preprocess/fit_parabolic_f_matlab/output/T_817_907/para_coef.bin");
    sim.read_comp_phad("/Users/yue/Dropbox/Research/codes/phasefield/Moelans_phase_field/preprocess/fit_parabolic_f_matlab/output/T_817_907/comp_phad.bin");
    
    // Restart from previous calculations
//    sim.read_init_cond((out_prefix + "phia_1000000.bin").c_str(), (out_prefix + "comp_1000000.bin").c_str());
//    sim.write_mem();
//    sim.restart(1000000); // set step counter
    
    // Start simulation steps
    sim.run();
    
    return 0;

}
