r"""
The fileio methods are derived from (d42430b):
https://github.com/jl-wynen/foldilearn/blob/master/foldilearn/fileio.py

"""
import json
import re
import isle
import dataclasses
from pathlib import Path
import yaml
from logging import getLogger
from isle.scripts.show import _nconfig,_overviewFigure,_loadConfigurationData, _loadMetadata

def read_params(path):
    r"""
    Reads parameters from a.json file
    :param path: path to parameter.json
    :return: dict, parameters for the simulation
    """

    with open(path, 'r') as file:
        param = json.load(file)

    param["sigmaKappa"] = -1
    param["hopping"] = isle.action.HFAHopping.EXP
    param["basis"] = isle.action.HFABasis.PARTICLE_HOLE
    param["algorithm"] = isle.action.HFAAlgorithm.DIRECT_SINGLE

    return param

def makeIsleParams(params):
    isle_params = {}
    for key in params.keys():
        if type(params[key]) is dict:
            for internalKey in params[key].keys():
                isle_params[key.replace(' ','')+"_"+internalKey.replace(' ','')] = params[key][internalKey]
        else:
            isle_params[key.replace(' ','')] = params[key]
        
    return isle.util.parameters(**isle_params)

def get_fn(preStr,posStr,params):
    r"""
        This Function Generalizes the file name. The general layout is
        preStr_{params}_{posStr}
        Typically you want posStr to be the file extension and preStr to identify the thing you are storing
    """
    fn = preStr + "_"
    
    for key in params.keys():
        # skip a few
        if key in ["sigmaKappa", "hopping", "basis", "algorithm", "lattice", "Nx"]:
            continue
            
        if type(params[key]) is dict:
            fn += f"{key.replace(' ','')}"
            for innerKey in params[key].keys():
                fn += f"-{innerKey.replace(' ','')}{params[key][innerKey]}"
            fn += "_"
        else:
            fn += f"{key.replace(' ','')}{params[key]}_"
            
    fn = fn[:-1] # remove last _
    fn += posStr
    
    return Path(fn)

def writeParamsH5(h5f,params):
    for key in params.keys():
        # skip a few
        if key in ["sigmaKappa", "hopping", "basis", "algorithm"]:
            continue
            
        if type(params[key]) is dict:
            for innerKey in params[key].keys():
                h5f.create_dataset( f"params/{key.replace(' ','')}/{innerKey.replace(' ','')}",
                    data = params[key][innerKey]
                )
        else:
            h5f.create_dataset( f"params/{key.replace(' ','')}",
                data = params[key]
            )

def make_action(lat, params):
    r"""
    Creates an isle action

    :param lat: lattice type
    :param params: isle.parameters
    :return: isle.SumAction (HubbardGaugeAction+HubbardFermiAction)
    """
    import isle.action

    return isle.action.HubbardGaugeAction(params.tilde("U", lat)) \
        + isle.action.makeHubbardFermiAction(lat,
                                             params.beta,
                                             params.tilde("mu", lat),
                                             params.sigmaKappa,
                                             params.hopping,
                                             params.basis,
                                             params.algorithm
        )

def overview(infname):
    """!
    Show an overview of a HDF5 file.
    """

    lattice, params, makeActionSrc = _loadMetadata(infname, isle.fileio.fileType(infname))

    log = getLogger("isle.show")
    log.info("Showing overview of file %s", infname)

    if lattice is None or params is None or makeActionSrc is None:
        log.error("Could not find all required information in the input file to generate an overview."
                  "Need HDF5 files.")
        return

    totalPhi, logWeights, trajPoints = _loadConfigurationData(infname)

    # set up the figure
    fig, (axTP, axAct, axPhase, axPhase2D, axPhi, axPhiHist, axText) = _overviewFigure()
    fig.canvas.set_window_title(f"Isle Overview - {infname}")

    # plot a bunch of stuff
    isle.plotting.plotTotalPhi(totalPhi, axPhi, axPhiHist)
    isle.plotting.plotTrajPoints(trajPoints, axTP)
    isle.plotting.plotWeights(logWeights, axAct)
    isle.plotting.plotPhase(logWeights, axPhase, axPhase2D)

    # show metadata at bottom of figure
    axText.axis("off")

    txt = rf"""
    """

    axText.text(0, 0, fontsize=13, linespacing=1, verticalalignment="bottom",
                s=txt
    )

    fig.tight_layout()

    return fig
