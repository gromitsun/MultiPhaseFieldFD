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

/******************** Actual reading functions ********************/

// Read input parameters
template <typename Type>
void readfile(const std::string & filename,
              unsigned int & nx, unsigned int & ny, unsigned int & nz, 
              Parameter<Type> & paras)
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
                paras.dx = (Type)std::stod(value);
            }
            else if (key == "dt")
            {
                paras.dt = (Type)std::stod(value);
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
                paras.nt = (unsigned int)std::stoul(value);
            }
            else if (key == "Da")
            {
                paras.Da = (Type)std::stod(value);
            }
            else if (key == "Db")
            {
                paras.Db = (Type)std::stod(value);
            }
            else if (key == "sigma")
            {
                paras.sigma = (Type)std::stod(value);
            }
            else if (key == "l")
            {
                paras.l = (Type)std::stod(value);
            }
            else if (key == "T_start")
            {
                paras.T_start = (Type)std::stod(value);
            }
            else if (key == "dT_dt")
            {
                paras.dT_dt = (Type)std::stod(value);
            }
            else if (key == "t_skip")
            {
                paras.t_skip = (unsigned int)std::stoul(value);
            }
            else if (key == "dT_recalc")
            {
                paras.dT_recalc = (Type)std::stod(value);
            }
            else if (key == "dT_data")
            {
                paras.dT_data = (Type)std::stod(value);
            }
            else if (key == "nT_data")
            {
                paras.nT_data = (Type)std::stoul(value);
            }
            else if (key == "T_start_data")
            {
                paras.T_start_data = (Type)std::stod(value);
            }
            else std::cout << key << " = " << value << " not understood!" << std::endl;
        }    
    }
    std::cout << "Done!" << std::endl;

    // Print inputs read in
    std::cout << "------- Parameters -------" << std::endl;
    
    std::cout << "*** Simulation parameters ***\n";
    std::cout << "nx = " << nx << std::endl;
    std::cout << "ny = " << ny << std::endl;
    std::cout << "nz = " << nz << std::endl;
    std::cout << "nt = " << paras.nt << std::endl;
    std::cout << "dx = " << paras.dx << std::endl;
    std::cout << "dt = " << paras.dt << std::endl;
    std::cout << "dT_recalc = " << paras.dT_recalc << std::endl;
    
    std::cout << "*** Parabolic coefficients parameters ***\n";
    std::cout << "dT_data = " << paras.dT_data << std::endl;
    std::cout << "nT_data = " << paras.nT_data << std::endl;
    std::cout << "T_start_data = " << paras.T_start_data << std::endl;
    
    std::cout << "*** Physical parameters ***\n";
    std::cout << "Da = " << paras.Da << std::endl;
    std::cout << "Db = " << paras.Db << std::endl;
    std::cout << "sigma = " << paras.sigma << std::endl;
    std::cout << "l = " << paras.l << std::endl;
    std::cout << "T_start = " << paras.T_start << std::endl;
    std::cout << "dT_dt = " << paras.dT_dt << std::endl;
    
    std::cout << "*** Output parameters ***\n";
    std::cout << "t_skip = " << paras.t_skip << std::endl;
    
    std::cout << "------- End -------" << std::endl;
}


// Read paths
void readfile(const std::string & filename, Settings & sets)
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
            if (key == "parameters_file")
            {
                sets.parameters_file = value;
            }
            else if (key == "init_phia")
            {
                sets.init_phia = value;
            }
            else if (key == "init_comp")
            {
                sets.init_comp = value;
            }
            else if (key == "comp_phad")
            {
                sets.comp_phad = value;
            }
            else if (key == "out_prefix")
            {
                sets.out_prefix = value;
            }
            else if (key == "log_file")
            {
                sets.log_file = value;
            }
            else if (key == "ndims")
            {
                sets.ndims = std::stoi(value);
            }
            else if (key == "restart")
            {
                sets.restart = (unsigned int)std::stoul(value);
            }
            else std::cout << key << " = " << value << " not understood!" << std::endl;
        }
    }
    std::cout << "Done!" << std::endl;
    
    // Print inputs read in
    std::cout << "------- Settings -------" << std::endl;
    
    std::cout << "*** Paths ***" << std::endl;
    std::cout << "parameters_file = " << sets.parameters_file << std::endl;
    std::cout << "init_phia = " << sets.init_phia << std::endl;
    std::cout << "init_comp = " << sets.init_comp << std::endl;
    std::cout << "comp_phad = " << sets.comp_phad << std::endl;
    std::cout << "out_prefix = " << sets.out_prefix << std::endl;
    std::cout << "log_file = " << sets.log_file << std::endl;
    
    std::cout << "*** Others ***" << std::endl;
    std::cout << "ndims = " << sets.ndims << std::endl;
    std::cout << "restart = " << sets.restart << std::endl;
    
    
    
    std::cout << "------- End -------" << std::endl;
}

/******************** Explicit instantiation ********************/
template void readfile<double>(const std::string & filename,
                               unsigned int & nx, unsigned int & ny, unsigned int & nz,
                               Parameter<double> & paras);

template void readfile<float>(const std::string & filename,
                              unsigned int & nx, unsigned int & ny, unsigned int & nz,
                              Parameter<float> & paras);



