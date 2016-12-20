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
#include "input.hpp"
#include "parameter_type.hpp"
#include "util.hpp"

int main(int argc, const char * argv[])
{
    std::cout << "Program starts\n";
    
    /* Read command line inputs */
    if (argc < 2)
    {
        std::cerr << "ERROR: No settings file specified!" << std::endl;
        exit(EXIT_FAILURE);
    }
    
    
    // Input filename
    std::string settings_file = argv[1];
    
    /* Read path settings */
    Settings sets;
    readfile(settings_file, sets);
    
    
//    if (sets.ndims == 2)
//    {
//        std::cout << "Starting 2D simulation ... \n";
//        
//        Simulator_2D<double> sim{};
//        
//        // Read input parameters
//        sim.read_input(sets.parameters_file.c_str());
//        
//        // Initialize program
//        sim.init_cl(CL_DEVICE_TYPE_GPU, 1);
//        
//        sim.init_sim(0, 0.001, sets.out_prefix.c_str());
//        
//        sim.build_kernel("kernel_double_2d.cl"); // contains memcopy host -> device
//        
//        // Read data from file
//        sim.read_init_cond(sets.init_phia.c_str(), sets.init_comp.c_str());
//        sim.write_mem();
//        sim.read_comp_phad(sets.comp_phad.c_str());
//        
//        // Restart from previous calculations
//        if (sets.restart)
//        {
//            sim.read_init_cond(time2fname(sets.out_prefix + "phia_", sets.restart).c_str(), time2fname(sets.out_prefix + "comp_", sets.restart).c_str());
//            sim.write_mem();
//            sim.restart(sets.restart); // set step counter
//        }
//        
//        // Start simulation steps
//        sim.run();
//        
//        return 0;
//    }
    
    std::cout << "Starting 3D simulation ... \n";

    Simulator_3D<double> sim{};
    
    // Read input parameters
    sim.read_input(sets.parameters_file.c_str());
    
    // Initialize program
    sim.init_cl(CL_DEVICE_TYPE_GPU, 1);
    
    sim.init_sim(0, 0.001, sets.out_prefix.c_str());
    
    sim.build_kernel("kernel_double_3d.cl"); // contains memcopy host -> device
    
    // Read data from file
    sim.read_init_cond(sets.init_phia.c_str(), sets.init_comp.c_str());
    sim.write_mem();
    sim.read_comp_phad(sets.comp_phad.c_str());
    
    // Restart from previous calculations
    if (sets.restart)
    {
        sim.read_init_cond(time2fname(sets.out_prefix + "phia_", sets.restart).c_str(), time2fname(sets.out_prefix + "comp_", sets.restart).c_str());
        sim.write_mem();
        sim.restart(sets.restart); // set step counter
    }
    
    // Start simulation steps
    sim.run();
    
    return 0;

}
