import isle
import numpy as np
import h5py as h5

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

def estimate_correlators(fn, Nconf, Nbst = 100):
    with h5.File(fn,'r') as h5f:
        actVals = h5f["weights/actVal"][()][:Nconf]
        
        try:
            logDetJ = h5f["weights/logdetJ"][()][:Nconf]
        except(KeyError):
            logDetJ = np.zeros_like(actVals)
            
        Corr = h5f["correlation_functions/single_particle/destruction_creation"][()][:Nconf,:,:,:]
        Corr[:,:,:,1:-1:-1] += h5f["correlation_functions/single_particle/creation_destruction"][()][:Nconf,:,:,1:-1:-1]
        Corr[:,:,:,1:-1:-1] /= 2
        
        _,Nx,_,Nt = Corr.shape
        
        weights = np.exp(-1j*(actVals-logDetJ).imag)
        
        Corr *= weights[:,None,None,None]
        
        Corr_est = np.zeros( shape=(Nbst, Nx, Nt), dtype = complex )
        Stat_est = np.zeros( Nbst )
        
        for k in range(Nbst):
            idx = np.random.randint( low=0,high=Nconf, size=Nconf)
            stat = weights[idx].mean(axis=0)
            Stat_est[k] = np.abs(stat)
            Corr_est[k,0,:] = Corr[idx,0,0,:].mean(axis=0)/stat
            Corr_est[k,1,:] = Corr[idx,1,1,:].mean(axis=0)/stat
        return {
            "Corr": [Corr_est.mean(axis=0), Corr_est.std(axis=0)],
            "StatPower": [Stat_est.mean(axis=0), Stat_est.std(axis=0)]
        }

def estimate_statistical_power(fn, Nconf, Nbst = 100):
    with h5.File(fn,'r') as h5f:
        actVals = h5f["weights/actVal"][()][:Nconf]
        
        try:
            logDetJ = h5f["weights/logdetJ"][()][:Nconf]
        except(KeyError):
            logDetJ = np.zeros_like(actVals)
        
        weights = np.exp(-1j*(actVals-logDetJ).imag)

        Stat_est = np.zeros( Nbst )
        
        for k in range(Nbst):
            idx = np.random.randint( low=0,high=Nconf, size=Nconf)

            Stat_est[k] = np.abs(weights[idx].mean(axis=0))

        return Stat_est.mean(axis=0), Stat_est.std(axis=0)
