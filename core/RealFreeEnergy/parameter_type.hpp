//
//  parameter_type.hpp
//  MoelansPhaseFieldFD
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
    Type a[5];
    Type b[6];
    // evolution equation coefficient
    Type m; // symmetric mobility
    Type L; // kinetic coefficient
    // other physical variables
    Type T;
    Type T_gibbs;       // temperature used in Gibbs free energy functions
    Type T_gibbs_next;
    Type RT; // R * T
    Type delta_comp_eq;
    Type compa_eq;
    Type compb_eq;
    Type f2a, f2b; // second derivatives at equilibrium compositions
    Type kba; // partition coefficient (f2a/f2b) at equilibrium compositions (c.f. Steinbach 2006 PRE)
};


#endif /* parameter_type_h */
