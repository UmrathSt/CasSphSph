"""worker.py provides functionality which is useful for the calculation
   of Casimir quantities by using the syntax:

   python worker.py config_filename.config result_filename.result
    
   if worker.py is called with a third parameter it is supposed to be
   a float and interpreted as center-to-center sphere-sphere distance.

"""

import configparser as ConfigParser #ConfigParser <- in Python 2.x
import sys
import geometry
import thermodynamics
import materials
import os
import numpy as np

args = sys.argv
if not len(args) in [3,4]:
    raise ValueError("Name of config file and output file has to be specified. \n"+
	                 "Specification of distance is optional.")
finput, foutput = args[1], args[2]

config = ConfigParser.RawConfigParser()
config.read(finput)

def assign_radius(cfg_ident_name):
    try:
        r = config.getfloat("geometry", cfg_ident_name)
    except (TypeError, ValueError):
        try: 
            r = list(map(float, config.get("geometry", cfg_ident_name).split(",")))
        except:
            raise TypeError("%s has to be either a single material instance, " %(cfg_ident_name)+
                            "or a list of material instances.")
    return r
    

def return_geometry():
    """reads the section [geometry] of the config-file and sets up an instance
       of geometry.SphereSphere. Materials occuring in the given geometry are
       specified in the section [materials] of the config-file

    """
    r1 = assign_radius("ra")
    r2 = assign_radius("rb")

    try: 
        L = float(args[3])
    except IndexError:
        L = config.getfloat("geometry", "d")
    try:
        precision = config.getfloat("geometry", "precision")
    except ValueError:
        if not precision == "automatic":
            print("precision has to be a float or \"automatic\"")
    try:
        lmax = config.getint("geometry", "lmax")
    except ValueError:
        lmax = config.get("geometry", "lmax")
        if lmax == "automatic":
            try:
                l_offset = config.getint("geometry", "l_offset")
            except ValueError:
                l_offset = None      
        if not lmax == "automatic":
            print("lmax has to be given as integer or \"automatic\"") 
    try:
        lssmax = config.getint("geometry", "lssmax")
    except ValueError:
        lssmax = config.get("geometry", "lssmax")
        if not lssmax == "automatic":
            print("lssmax has to be given as integer or \"automatic\"") 
    try: 
        l_offset = config.getint("geometry","l_offset")
    except ConfigParser.NoOptionError:
        l_offset = 0        
    try: 
        evaluation = config.get("geometry","evaluation")
    except ConfigParser.NoOptionError:
        print("EXCEPTION RAISED")
        evaluation = "lndet"
    mat1 = eval(config.get("materials", "mata"))
    mat2 = eval(config.get("materials", "matb"))
    mat3 = eval(config.get("materials", "matmd"))
    if type(mat1) == tuple:
        mat1 = list(mat1)
    if type(mat2) == tuple:
        mat2 = list(mat2)
    G = geometry.SphereSphere(r1=r1, r2=r2, L=L, mat1=mat1, 
                    mat2=mat2, mat3=mat3, lmax=lmax, l_offset=l_offset, 
            lssmax=lssmax, precision=precision, forceFlag=0, evaluation=evaluation)
    return G
geom = return_geometry()

def do_calculation():
    """Sets up the calculation of thermodynamic quantities specified in the
       section [thermodynamics] of the config-file for a given geometry 
       specified in the section [geometry] of the config-file.

    """
    dT = config.getfloat("thermodynamics", "Tmin")
    try:
        nmax = config.getint("thermodynamics", "nmax")
    except ValueError:
        nmax = config.get("thermodynamics", "nmax")
        if not nmax == "automatic":
            print("nmax has to be given as integer or \"automatic\"") 
    try:
        analytic_n0 = config.get("geometry", "analytic_n0")
    except ValueError:
        print("analytic_n0 is not given correctly: ", analytic_n0)
    analytic_n0 = True
    if type(geom.r1) == list or type(geom.r2) == list:
        analytic_n0 = False
    Tmax = config.getfloat("thermodynamics", "Tmax")
    f = thermodynamics.finiteT(dT, nmax, Tmax, geom, analytic_n0)
    quantity = config.get("thermodynamics", "quantity")
    result = eval("f.%s" %quantity)
    return result, f
quantity = config.get("thermodynamics", "quantity")
result, fclass = do_calculation()

def job_infos():
    """collects relevant information on the done calculation 
       and prefixes it as a comment to the result-file.

    """
    mat1 = config.get("materials", "mata")
    mat2 = config.get("materials", "matb")
    matMd = config.get("materials", "matmd")
    log_data = ("# information on job: %s \n" %(finput)+
                "# r1 = %s, r2 = %s, d = %s \n" %(geom.r1, geom.r2, geom.L)+
                "# mat1 = %s, mat2 = %s, matMd = %s \n" %(mat1, mat2, matMd)+
                "# lmax = %i, l_offset = %i, lssmax = %i, nmax = %i, deltaT= %.4f, Tmax = %.4f \n" %(geom.lmax, geom.l_offset, geom.lssmax, 
                                                        fclass.nmax, fclass.deltaT, fclass.Tmax)+
                "# calculated quantity: %s \n" %(config.get("thermodynamics", "quantity"))+
                "# type of evaluation: %s \n" %(geom.evaluation)+
                "# precision: %s \n" %(geom.precision))
    if geom.evaluation == "trace":
        log_data+=("# order of the trace expansion (number of round-trips) for various m-subspaces.\n"
                  +"# The Format is: m-subspace: order \n"+
                   "# %s \n" %str(geom.roundTrip_dct))
    return log_data

def save_result(result, foutput):
    """saves the desired results to the specified output file which 
       was initially passed as the second command-line argument.

    """
    if os.path.exists(foutput):
        foutput += "1"
    outfile = open(foutput, "w")
    comment = job_infos()
    outfile.write(comment)
    outfile.close()
    outfile = open(foutput, "ab")
    np.savetxt(outfile, result, delimiter=",")
    outfile.close()

save_result(result, foutput)
    


