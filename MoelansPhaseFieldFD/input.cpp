/*input.cpp*/
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include "input.hpp"


/******************** String processing functions ********************/
inline std::string trim_comment(const std::string & s, const std::string & delimiter="#")
{
    if (s.empty())
        return s;
    else
        return s.substr(0, s.find(delimiter));
}

inline std::string trim_right(
  const std::string & s,
  const std::string & delimiters = " \f\n\r\t\v" )
{
	if (s.empty())
		return s;
	return s.substr(0, s.find_last_not_of(delimiters) + 1);
}

inline std::string trim_left(
  const std::string & s,
  const std::string & delimiters = " \f\n\r\t\v" )
{
	if (s.empty())
		return s;
	return s.substr(s.find_first_not_of(delimiters));
}

inline std::string trim(
  const std::string & s,
  const std::string & delimiters = " \f\n\r\t\v" )
{
	return trim_left(trim_right(trim_comment(s), delimiters), delimiters);
}

inline bool parse_parameter(const std::string & line, std::string & key, std::string & value, const std::string & sep = "=")
{
	std::size_t pos = line.find(sep);
	if (pos == std::string::npos)
		return false;
	else
	{
		key = trim(line.substr(0, pos));
		value = trim(line.substr(pos+1));
		return true;
	}
}


/*****************************************************************/

/******************** Actual reading function ********************/

template <typename Type>
void readfile(const std::string & filename,
              unsigned int & nx, unsigned int & ny, unsigned int & nz, unsigned int & nt,
              Type & dx, Type & dt,
              Type & Da, Type & Db, Type & sigma, Type & l, Type & T_start, Type & dT_dt,
              unsigned int & t_skip,
              Type & PHI_MIN, Type & COMP_MIN, Type & T_MIN,
              Type & PHI_INC, Type & COMP_INC, Type & T_INC,
              size_t & PHI_NUM, size_t & COMP_NUM, size_t & T_NUM)
{
	std::ifstream fin;
	std::string s;
	fin.open(filename.c_str());

    if (fin.fail())
    {
        std::cout << "Failed to open file " << filename << "..." << std::endl;
        exit(-1);
    }

	std::string key, value;
	std::cout << "Reading inputs from " << filename << "..." << std::endl;
	while (getline(fin, s))
	{
		if (!trim(s).empty() && parse_parameter(s, key, value))
		{
			if (key == "dx")
			{
				dx = (Type)std::stod(value);
			}
			else if (key == "dt")
			{
				dt = (Type)std::stod(value);
			}
			else if (key == "nx")
			{
				nx = (unsigned int)std::stoul(value);
			}
			else if (key == "ny")
			{
				ny = (unsigned int)std::stoul(value);
			}
            else if (key == "nz")
            {
                nz = (unsigned int)std::stoul(value);
            }
			else if (key == "nt")
			{
				nt = (unsigned int)std::stoul(value);
			}
			else if (key == "Da")
			{
				Da = (Type)std::stod(value);
			}
			else if (key == "Db")
			{
				Db = (Type)std::stod(value);
			}
			else if (key == "sigma")
			{
				sigma = (Type)std::stod(value);
			}
			else if (key == "l")
			{
				l = (Type)std::stod(value);
			}
            else if (key == "T_start")
            {
                T_start = (Type)std::stod(value);
            }
            else if (key == "dT_dt")
            {
                dT_dt = (Type)std::stod(value);
            }
			else if (key == "t_skip")
			{
				t_skip = (unsigned int)std::stoul(value);
			}
            else if (key == "PHI_MIN")
            {
                PHI_MIN = (Type)std::stod(value);
            }
            else if (key == "COMP_MIN")
            {
                COMP_MIN = (Type)std::stod(value);
            }
            else if (key == "T_MIN")
            {
                T_MIN = (Type)std::stod(value);
            }
            else if (key == "PHI_INC")
            {
                PHI_INC = (Type)std::stod(value);
            }
            else if (key == "COMP_INC")
            {
                COMP_INC = (Type)std::stod(value);
            }
            else if (key == "T_INC")
            {
                T_INC = (Type)std::stod(value);
            }
            else if (key == "PHI_NUM")
            {
                PHI_NUM = (unsigned int)std::stoul(value);
            }
            else if (key == "COMP_NUM")
            {
                COMP_NUM = (unsigned int)std::stoul(value);
            }
            else if (key == "T_NUM")
            {
                T_NUM = (unsigned int)std::stoul(value);
            }
			else std::cout << key << " = " << value << " not understood!" << std::endl;
		}	
	}
	std::cout << "Done!" << std::endl;

	// Print inputs read in
	std::cout << "------- Inputs -------" << std::endl;
    
    std::cout << "*** Simulation parameters ***\n";
	std::cout << "nx = " << nx << std::endl;
	std::cout << "ny = " << ny << std::endl;
    std::cout << "nz = " << nz << std::endl;
	std::cout << "nt = " << nt << std::endl;
	std::cout << "dx = " << dx << std::endl;
	std::cout << "dt = " << dt << std::endl;
    
    std::cout << "*** Physical parameters ***\n";
	std::cout << "Da = " << Da << std::endl;
	std::cout << "Db = " << Db << std::endl;
	std::cout << "sigma = " << sigma << std::endl;
	std::cout << "l = " << l << std::endl;
    std::cout << "T_start = " << T_start << std::endl;
    std::cout << "dT_dt = " << dT_dt << std::endl;
    
    std::cout << "*** Output parameters ***\n";
	std::cout << "t_skip = " << t_skip << std::endl;
    
    std::cout << "*** Interpolation parameters ***\n";
    std::cout << "PHI_MIN = " << PHI_MIN << std::endl;
    std::cout << "COMP_MIN = " << COMP_MIN << std::endl;
    std::cout << "T_MIN = " << T_MIN << std::endl;
    std::cout << "PHI_INC = " << PHI_INC << std::endl;
    std::cout << "COMP_INC = " << COMP_INC << std::endl;
    std::cout << "T_INC = " << T_INC << std::endl;
    std::cout << "PHI_NUM = " << PHI_NUM << std::endl;
    std::cout << "COMP_NUM = " << COMP_NUM << std::endl;
    std::cout << "T_NUM = " << T_NUM << std::endl;
    
	std::cout << "------- End -------" << std::endl;
}


/******************** Explicit instantiation ********************/
template void readfile<double>(const std::string & filename,
                               unsigned int & nx, unsigned int & ny, unsigned int & nz, unsigned int & nt,
                               double & dx, double & dt,
                               double & Da, double & Db, double & sigma, double & l, double & T_start, double & dT_dt,
                               unsigned int & t_skip,
                               double & PHI_MIN, double & COMP_MIN, double & T_MIN,
                               double & PHI_INC, double & COMP_INC, double & T_INC,
                               size_t & PHI_NUM, size_t & COMP_NUM, size_t & T_NUM);

template void readfile<float>(const std::string & filename,
                              unsigned int & nx, unsigned int & ny, unsigned int & nz, unsigned int & nt,
                              float & dx, float & dt,
                              float & Da, float & Db, float & sigma, float & l, float & T_start, float & dT_dt,
                              unsigned int & t_skip,
                              float & PHI_MIN, float & COMP_MIN, float & T_MIN,
                              float & PHI_INC, float & COMP_INC, float & T_INC,
                              size_t & PHI_NUM, size_t & COMP_NUM, size_t & T_NUM);



