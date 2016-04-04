/*input.hpp*/
#ifndef INPUT_HPP
#define INPUT_HPP

#include <string>


template <typename Type>
void readfile(const std::string & filename,
              unsigned int & nx, unsigned int & ny, unsigned int & nz, unsigned int & nt,
              Type & dx, Type & dt,
              Type & Da, Type & Db, Type & sigma, Type & l, Type & T_start, Type & dT_dt,
              unsigned int & t_skip,
              Type & PHI_MIN, Type & COMP_MIN, Type & T_MIN,
              Type & PHI_INC, Type & COMP_INC, Type & T_INC,
              size_t & PHI_NUM, size_t & COMP_NUM, size_t & T_NUM);

#endif