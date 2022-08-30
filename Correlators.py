import isle
import numpy as np

import isle
import isle.drivers
import isle.meas

def measure_correlators(fn,params):
    measState = isle.drivers.meas.init(fn, fn, True)

    params = measState.params

    lat = measState.lattice

    assert params.basis == isle.action.HFABasis.PARTICLE_HOLE
    assert params.hopping == isle.action.HFAHopping.EXP

    hfm = isle.HubbardFermiMatrixExp(
        params.tilde(measState.lattice.hopping(), lat.nt()),
        params.tilde("mu", lat),
        params.sigmaKappa
    )

    allToAll = {s: isle.meas.propagator.AllToAll(hfm, s) for s in isle.Species.values()}
    _, diagonalize = np.linalg.eigh(isle.Matrix(hfm.kappaTilde()))

    measurements = [
        # collect all weights and store them in consolidated datasets instead of
        # spread out over many HDF5 groups
        isle.meas.CollectWeights("weights"),
        # single particle correlator for particles / spin up
        isle.meas.SingleParticleCorrelator(allToAll[isle.Species.PARTICLE],
                                           lat,
                                           "correlation_functions/single_particle",
                                           transform=diagonalize
                                           )
    ]

    # Run the measurements on all configurations in the input file.
    # This automatically saves all results to the output file when done.
    measState(measurements)
