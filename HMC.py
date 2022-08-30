import isle
import isle.action
import isle.drivers
import torch
from logging import getLogger

from Misc import get_fn, make_action
import numpy as np 
def HMC(param):
    log = getLogger(f"HMC")

    isle_lat = isle.LATTICES[param['lattice']]
    isle_lat.nt(param['nt'])

    isle_rng = isle.random.NumpyRNG(param['rng_seed'])
    isle_hmcState = isle.drivers.hmc.newRun(
        isle_lat,
        param,
        isle_rng,
        make_action,
        get_fn('HMC_Simulation','h5', param),
        False
    )

    phi = isle.Vector(
        isle_rng.normal(
            0,
            param.tilde("U", isle_lat)**(1/2),
            isle_lat.lattSize()
        ) + 1j*param['tangent_plane']
    )

    isle_evStage = isle.evolver.EvolutionStage(
        phi,
        isle_hmcState.action.eval(phi),
    )

    log.info("Thermalizing")
    isle_evolver = isle.evolver.ConstStepLeapfrog(
        action = isle_hmcState.action,
        length = param['MD_trajectory_length'],
        nstep = param['MD_steps'],
        rng = isle_rng
    )
    isle_evStage = isle_hmcState(isle_evStage, isle_evolver, param['bTherm'], saveFreq=0, checkpointFreq=0)
    isle_hmcState.resetIndex()

    log.info("Producing")
    isle_evolver = isle.evolver.ConstStepLeapfrog(
        action = isle_hmcState.action,
        length = param['MD_trajectory_length'],
        nstep = param['MD_steps'],
        rng = isle_rng,
    )
    isle_hmcState(
        isle_evStage,
        isle_evolver,
        param['nTraj'],
        saveFreq=param['sTherm'],
        checkpointFreq=10
    )


def ML_HMC(torch_model,param):
    log = getLogger(f"ML HMC")

    try:
        isle_lat = isle.LATTICES[param['lattice']]
    except:
        isle_lat = isle.LATTICES.loadExternal(param['lattice'])
    isle_lat.nt(param['nt'])

    isle_rng = isle.random.NumpyRNG(param['rng_seed'])
    isle_hmcState = isle.drivers.hmc.newRun(
        isle_lat,
        param,
        isle_rng,
        make_action,
        get_fn('MLHMC_Simulation','h5', param),
        False
    )

    Nt = param['nt']
    Nx = param['volume']//param['nt']
    model =  isle.evolver.transform.TorchTransform(torch_model, isle_hmcState.action, desired_shape = (Nt,Nx) )

    phi = isle.Vector(
        isle_rng.normal(
            0,
            param.tilde("U", isle_lat)**(1/2),
            isle_lat.lattSize()
        ) + 1j*param['tangent_plane']
    )

    phiT, actVal, logDetJ = model.forward(phi)
    isle_evStage = isle.evolver.EvolutionStage(
        phi=phiT,
        actVal=actVal,
        trajPoint=1,
        logWeights={"logdetJ": logDetJ},
        phi_RTP = phi # setting this allows to store the real-/tangentplane config used in ConstStepLeapfrogML
    )

    log.info("Thermalizing")

    isle_evolver = isle.evolver.ConstStepLeapfrogML(
        action = isle_hmcState.action,
        length = param['MD_trajectory_length'],
        nstep = param['MD_steps'],
        rng = isle_rng,
        transform = model
    )
    isle_evStage = isle_hmcState(isle_evStage, isle_evolver, param['bTherm'], saveFreq=0, checkpointFreq=0)
    isle_hmcState.resetIndex()

    log.info("Producing")
    isle_evolver = isle.evolver.ConstStepLeapfrogML(
        action = isle_hmcState.action,
        length = param['MD_trajectory_length'],
        nstep = param['MD_steps'],
        rng = isle_rng,
        transform = model
    )
    isle_hmcState(
        isle_evStage,
        isle_evolver,
        param['nTraj'],
        saveFreq=param['sTherm'],
        checkpointFreq=10
    )
