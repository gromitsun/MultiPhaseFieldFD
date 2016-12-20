//
//  parameter_type.hpp
//  SteinbachPhaseFieldFD
//
//  Created by Yue Sun on 6/8/16.
//  Copyright © 2016 Yue Sun. All rights reserved.
//

#ifndef parameter_type_h
#define parameter_type_h


template <typename Type>
struct Parameter
{
    /* Input parameters */
    // physical parameters
    Type Da;
    Type Db;
    Type sigma;
    Type l;
    Type T;
    Type dT_dt;
    Type T_start;
    // phase field parameters
    Type kappa;    // = 0.75*sigma*l
    Type mk;       // = 6*sigma/l
    Type L0;       // L = L0/zeta
    // simulation parameters
    Type dx;
    Type dt;
    unsigned int nt;
    unsigned int t_skip; // output every t_skip steps
    Type dT_recalc;
    // parabolic data file info
    Type dT_data;
    unsigned int nT_data;
    Type T_start_data;    
};

template <typename Type>
struct Variable
{
    // parabolic coefficients of the free energy functions
    Type a2;
    Type a1;
    Type a0;
    Type b2;
    Type b1;
    Type b0;
    // evolution equation coefficient
    Type m; // symmetric mobility
    Type L; // kinetic coefficient
    Type Ma, Mb; // mobilities of a and b (= Da/a2 ,Db/b2)
    // other physical variables
    Type T;
    Type T_gibbs;       // temperature used in Gibbs free energy functions
    Type T_gibbs_next;
    Type delta_comp_eq;
    Type compa_eq;
    Type compb_eq;
};


#endif /* parameter_type_h */
