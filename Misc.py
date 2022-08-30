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


# Regex pattern to extract information from a file name.
FNAME_PATTERN = re.compile(r"^(?P<kind>\w+)"  # file kind
                           r"\.(?P<latname>\w+)"  # latname
                           r"\.(?P<hopping>\w+)"  # hopping
                           r"\.nt(?P<nt>\d+)"         # nt
                           r"\.beta(?P<beta>[\d_]+)"  # beta
                           r"\.U(?P<U>[\d_]+)"  # U
                           r"\.mu(?P<mu>[\d_]+)"
                           r"(?:\.(?P<extra>\w+))?$")  # extra stuff


def formatFloat(x):
    r"""
    Format a single float for file names.
    """

    return f"{float(x)}".replace(".", "_")


def get_fn(prepend, end, param, path = "Results/"):
    OUTFILE_FMT = "{kind}.flow{flow}.{latname}.{hopping}.nt{nt}.beta{beta}.U{U}.mu{mu}.lr{lr}.epochs{epochs}.minibatches{mb}.numLayer{num_layers}.{end}"

    try:
        lat = isle.LATTICES[param['lattice']]
    except:
        lat = isle.LATTICES.loadExternal(param['lattice'])
    lat.nt(param['nt'])

    fn = path

    fn+= OUTFILE_FMT.format(
        kind = prepend,
        flow = formatFloat(param.tau_f),
        latname = lat.name.replace(" ", "_") if isinstance(lat, isle.Lattice) else lat,
        hopping="exp" if param.hopping == isle.action.HFAHopping.EXP else "dia",
        nt=param.nt,
        beta=formatFloat(param.beta),
        U=formatFloat(param.U),
        mu=formatFloat(param.mu),
        lr=formatFloat(param.lr),
        epochs=param.epochs,
        mb = param.mini_batches,
        num_layers = param.number_layers,
        end = end
    )

    return Path(fn)


def read_params(path):
    r"""
    Reads parameters from a.json file
    :param path: path to parameter.json
    :return: dict, parameters for the simulation
    """

    with open(path, 'r') as file:
        param = json.load(file)

    for key in param.keys():
        if isinstance(param[key], list):
            param[key] = tuple(param[key])

    # now fill in a few things
    if param['lattice'] == "two_sites":
        param['volume'] = 2*param['nt']
    elif param['lattice'] == "four_sites":
        param['volume'] = 4*param['nt']
    elif param['lattice'] == "eight_sites":
        param['volume'] = 8*param['nt']
    elif param['lattice'] == "18_sites":
        param['volume'] = 18*param['nt']
    else:
        raise RuntimeError(f"Volume not implemented for: {param['lattice']}")

    param["sigmaKappa"] = -1
    param["hopping"] = isle.action.HFAHopping.EXP
    param["basis"] = isle.action.HFABasis.PARTICLE_HOLE
    param["algorithm"] = isle.action.HFAAlgorithm.DIRECT_SINGLE

    return isle.util.parameters(**param)


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
* Ensemble: $N_{{\mathrm{{config}}}}={_nconfig(infname)}$ 
* Lattice : {lattice.name}   $N_t={lattice.nt()}$, $N_x={lattice.nx()}$ '{lattice.comment}'
* Params  : $\beta = {params.beta}$, $U = {params.U}$, $\mu = {params.mu}$
* Model: $\lambda_{{lr}} = {params.lr}$, $epochs = {params.epochs}$, $mini\ batches = {params.mini_batches}$, {params.number_layers} PRACL
"""

    axText.text(0, 0, fontsize=13, linespacing=1, verticalalignment="bottom",
                s=txt
    )

    fig.tight_layout()

    return fig
