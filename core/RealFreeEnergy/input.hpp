/*input.hpp*/
#ifndef INPUT_HPP
#define INPUT_HPP

#include <string>
#include "parameter_type.hpp"

template <typename Type>
void readfile(const std::string & filename,
              unsigned int & nx, unsigned int & ny, unsigned int & nz, 
              Parameter<Type> & paras);

void readfile(const std::string & filename, Settings & sets);

#endif