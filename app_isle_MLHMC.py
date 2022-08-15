import torch
from lib_tensorSpecs import * # specification for the torch tensors

import numpy as np

from numpy import s_

import matplotlib.pyplot as plt

import h5py as h5

from logging import getLogger

import isle 

import isle.drivers

import isle.meas

from pathlib import Path

import lib_layer as Layers
import lib_loss as Losses

### Specify input / output files

# Name of the lattice.
LATTICE = "two_sites"

NCONF = 1000
Nt = 16
Nx = 2

tangentPlaneOffset = -4.99933002e-01

TRAJECTORY_LENGTH = 0.1

NUMBER_MD_STEPS = 10

BURNIN = 1000

THERMALIZATION = 10

NETWORKFILE = "NN equivNt_16initEps_0.001trainEps_1e-05trainBatchSize_10000trainMiniBatchSize_5000LR_1e-07 numPRCLLayers_1numIntLayers_8activ_Softsign()lossfn__class 'lib_loss.MinimizeImaginaryPartLoss'_epochs_1800.pt"

NETWORKSPECS = {
          "numPRCLLayers": 1,
          "numInternalLayers": 8,
          "activation": torch.nn.Softsign,
          "initEpsilon": 0.001,
          "trainEpsilon":  1e-05,
          "trainBatchSize": 10000,
          "trainMiniBatchSize": 5000,
          "lossFct": Losses.MinimizeImaginaryPartLoss,
          "learningRate": 1e-07,
          "Nt": 16,
        }

PARAMS = isle.util.parameters(
    beta=4,         # inverse temperature
    U=4,            # on-site coupling
    mu=3,           # chemical potential
    sigmaKappa=-1,  # prefactor of kappa for holes / spin down
                    # (+1 only allowed for bipartite lattices)

    # Those three control which implementation of the action gets used.
    # The values given here are the defaults.
    # See documentation in docs/algorithm.
    hopping=isle.action.HFAHopping.EXP,
    basis=isle.action.HFABasis.PARTICLE_HOLE,
    algorithm=isle.action.HFAAlgorithm.DIRECT_SINGLE
)

def makeAction(lat, params):
    # Import everything this function needs so it is self-contained.
    import isle
    import isle.action

    return isle.action.HubbardGaugeAction(params.tilde("U", lat)) \
        + isle.action.makeHubbardFermiAction(lat,
                                             params.beta,
                                             params.tilde("mu", lat),
                                             params.sigmaKappa,
                                             params.hopping,
                                             params.basis,
                                             params.algorithm)

def coupling_factory(numInternalLayers, internalLayer_factory, activation_factory, initEpsilon, **internalLayerKwargs):
    r"""
        \param:
            - numInternalLayers 
            - internalLayer_factory 
            - activation_factory 
            - initEpsilon 
            - internalLayerKwargs
        This factory generates an affine coupling layer (see lib_layer.py)
        using numInternalLayers of torch.nn.Modules created by internalLayer_factory 
        an seperated by activation_factory. 
        Depending on the number of numInternalLayers this generates a coupling
        of the form 
        ```
        AffineCoupling:
            m = Sequential([ m_layer1, activation, m_layer2, activation, ..., m_layerN ]),
            a = Sequential([ a_layer1, activation, a_layer2, activation, ..., a_layerN ]),
        ```
    """
    
    def generate_parameters():
        r"""
            This generates the list of internal layers used for the networks 
            called a,m in the affine coupling
            It initializes the parameters of these layes using a uniform 
            distribution between -initEpsilon and +initEpsilon
        """
        layerList = []
        for _ in range(numInternalLayers-1):
            # create internal layers using the factory
            layer = internalLayer_factory(**internalLayerKwargs)
            
            # initialize the parameters using a uniform distribution
            for param in layer.parameters():
                torch.nn.init.uniform_(param,-initEpsilon,initEpsilon)

            layerList.append(layer)
            
           # append a activation function
            layerList.append(activation_factory())

        # generate and initialize a last layer which is not followed by 
        # an activation function
        layer = internalLayer_factory(**internalLayerKwargs)
        for param in layer.parameters():
            torch.nn.init.uniform_(param,-initEpsilon,initEpsilon)
        
        layerList.append(layer)

        # put everything in a torch sequential container to apply 
        # the layers after each other
        return torch.nn.Sequential(*layerList)
    
    return Layers.AffineCoupling(
        m = generate_parameters(),
        a = generate_parameters()  
    )

def NN_factory(numPRCLLayers,coupling_factory,**couplingKwargs):
    r"""
        \param:
            - numPRCLLayers:   int, number of PRCL layers stored in the module
            - coupling_factory: callable, generating a coupling for the PRCL layer (both couplings in PRCL are generated in the same way) 
            - couplingKwargs: keyworded arguments for the coupling_factory
        This factory function creates a `numPRCLLayers` PRCL layer using 
        coupling(1,2) created by the coupling_factory. The resulting torch 
        module could look something like
        ```
        Sequential:
            PRCL(1):
                coupling1 = AffineCoupling(
                    m = LinearTransformation(Nt,Nx),
                    a = LinearTransformation(Nt,Nx)
                ),
                coupling2 = AffineCoupling(
                    m = LinearTransformation(Nt,Nx),
                    a = LinearTransformation(Nt,Nx)
                )
            PRCL(2):
                coupling1 = AffineCoupling(
                    m = LinearTransformation(Nt,Nx),
                    a = LinearTransformation(Nt,Nx)
                ),
                coupling2 = AffineCoupling(
                    m = LinearTransformation(Nt,Nx),
                    a = LinearTransformation(Nt,Nx)
                )
        ```
        The form of the coupling(1,2) varies depending on the coupling 
        factory.
    """

    # Store all PRCL layers in one list, later passed to the Sequential container
    NN = []

    for _ in range(numPRCLLayers):
        # Create a PRCL layer with couplings defined by the coupling factories
        NN.append(
            Layers.PRCL(
                Nt,Nx,
                coupling1 = coupling_factory(**couplingKwargs),
                coupling2 = coupling_factory(**couplingKwargs)
            )
        )

    return Layers.Sequential(NN)

def createNetwork():
    NN = NN_factory(
        numPRCLLayers = NETWORKSPECS["numPRCLLayers"],
        coupling_factory = coupling_factory,
        # coupling Kwargs
        numInternalLayers = NETWORKSPECS["numInternalLayers"], 
        internalLayer_factory = Layers.LinearTransformation, 
        activation_factory = NETWORKSPECS["activation"], 
        initEpsilon = NETWORKSPECS["initEpsilon"], 
        # Linear Transformation Args
        Nt = Nt,
        Nx = Nx//2
    )
        
    NN = Layers.Equivariance(NN)
    
    NN.load_state_dict(torch.load(NETWORKFILE))

    return NN

def MLHMC(args):

    # Get a logger. Use this instead of print() to output any and all information.
    log = getLogger("HMC")

    # Load the spatial lattice.
    # Note: This command loads a lattice that is distributed together with Isle.
    #       In order to load custom lattices from a file, use
    #       either  isle.LATTICES.loadExternal(filename)
    #       or  isle.fileio.yaml.loadLattice(filename)
    lat = isle.LATTICES[LATTICE]
    # Lattice files usually only contain information on the spatial lattice
    # to be more flexible. Set the number of time slices here.
    lat.nt(Nt)

    # Set up a random number generator.
    rng = isle.random.NumpyRNG(1075)

    # Set up a fresh HMC driver.
    # It handles all HMC evolution as well as I/O.
    # Last argument forbids the driver to overwrite any existing data.
    hmcState = isle.drivers.hmc.newRun(lat, PARAMS, rng, makeAction,
                                       args.outfile, False)

    # Generate a random initial condition.
    # Note that configurations must be vectors of complex numbers.
    phi = isle.Vector(rng.normal(0,
                                 PARAMS.tilde("U", lat)**(1/2),
                                 lat.lattSize())
                      +1j*tangentPlaneOffset)

    NN = createNetwork()
    NN = isle.evolver.transform.TorchTransform(NN, hmcState.action, (Nt,Nx))
    phiM,actVal,logDetJ = NN.forward(phi)

    # Run thermalization.
    log.info("Thermalizing")
    # The number of steps (99) must be one less than the number of trajectories below.
    evolver = isle.evolver.ConstStepLeapfrogML(action=hmcState.action, length=TRAJECTORY_LENGTH, nstep=NUMBER_MD_STEPS, rng=rng, transform=NN)

    evStage = isle.evolver.EvolutionStage(
        phi=phiM,
        actVal=actVal,
        trajPoint=1,
        logWeights={"logdetJ": logDetJ},
        phi_RTP = phi # setting this allows to store the real-/tangentplane config used in ConstStepLeapfrogML
    )

    # Thermalize configuration for 100 trajectories without saving anything.
    evStage = hmcState(evStage, evolver, BURNIN, saveFreq=0, checkpointFreq=0)
    # Reset the internal counter so we start saving configs at index 0.
    hmcState.resetIndex()

    # Run production.
    log.info("Producing")
    # Pick a new evolver with a constant number of steps to get a reproducible ensemble.
    evolver = isle.evolver.ConstStepLeapfrogML(action=hmcState.action, length=TRAJECTORY_LENGTH, nstep=NUMBER_MD_STEPS, rng=rng, transform=NN)
    # Produce configurations and save in intervals of 2 trajectories.
    # Place a checkpoint every 10 trajectories.
    hmcState(evStage, evolver, NCONF, saveFreq=THERMALIZATION, checkpointFreq=10)

    # That is it, clean up happens automatically.


def measurements(args):
    # Set up a measurement driver to run the measurements.
    measState = isle.drivers.meas.init(args.infile, args.outfile, args.overwrite)
    # The driver has retrieved all previously stored parameters from the input file,
    params = measState.params
    # as well as the lattice including the number of time slices nt.
    lat = measState.lattice

    # For simplicity do not allow the spin basis.
    assert params.basis == isle.action.HFABasis.PARTICLE_HOLE

    # Get "tilde" parameters (xTilde = x*beta/Nt) needed to construct measurements.
    muTilde = params.tilde("mu", lat)
    kappaTilde = params.tilde(measState.lattice.hopping(), lat.nt())

    # This object is a lower level interface for the Hubbard fermion action
    # needed by some measurements. The discretization (hopping) needs
    # to be selected manually.
    if params.hopping == isle.action.HFAHopping.DIA:
        hfm = isle.HubbardFermiMatrixDia(kappaTilde, muTilde, params.sigmaKappa)
    else:
        hfm = isle.HubbardFermiMatrixExp(kappaTilde, muTilde, params.sigmaKappa)

    # Define measurements to run.
    allToAll = {s: isle.meas.propagator.AllToAll(hfm, s) for s in isle.Species.values()}

    _, diagonalize = np.linalg.eigh(isle.Matrix(hfm.kappaTilde()))

    measurements = [
        # single particle correlator for particles / spin up
        isle.meas.SingleParticleCorrelator(allToAll[isle.Species.PARTICLE],
                                           lat,
                                           "correlation_functions/single_particle",
                                           configSlice=s_[::THERMALIZATION],
                                           transform=diagonalize)
    ]

    # Run the measurements on all configurations in the input file.
    # This automatically saves all results to the output file when done.
    measState(measurements)


if __name__ == "__main__":
    # Initialize Isle.
    # This sets up the command line interface, defines a barebones argument parser,
    # and parses and returns parsed arguments.
    # More complex parsers can be automatically defined or passed in manually.
    # See, e.g., `hmcThermalization.py` or `measure.py` examples.
    parser = isle.cli.makeDefaultParser(defaultLog="MLHMC.log",
                                        description="MLHMC")

    parser.add_argument("outfile", help="Output file", type=Path)
    parser.add_argument("--infile", help="Input file",type=Path, required = False)
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing output file.")
        
    args = isle.initialize(parser)

    if args.infile is None:
        args.infile = args.outfile

    MLHMC(args)

    measurements(args)





# ========================================================================


#Nx = 2
#
#fn = "1_configs.h5"
#
#with h5.File(fn,'r') as h5f:
#    S_eff  = h5f['S_eff'][()]
#    S      = h5f['S'][()]
#    phis_M = h5f['configsM'][()]
#    phis_R = h5f['configsR'][()]
#
#Csp = correlator(phis_M)
#Nconf,_,_,_ = Csp.shape
#Nbst = 10
#
#Csp_bst = torch.zeros((Nbst,Nt,Nx,Nx),dtype = torch.cdouble)
#for k in range(Nbst):
#    idx = torch.randint(low=0,high=Nconf,size=(Nconf,))
#    sample_C = Csp[idx]
#    sample_S = torch.exp(1j*S_eff[idx].imag)
#
#    Csp_bst[k] = (sample_C*sample_S[:,None,None,None]).mean(dim=0)/sample_S.mean(dim=0)
#
#Csp_est = Csp_bst.mean(dim = 0).numpy()
#Csp_err = Csp_bst.std(dim = 0).numpy()
#t = np.arange(Nt)*delta 
#
#plt.errorbar(t,Csp_est[:,0,0],Csp_err[:,0,0],capsize = 2)
#plt.yscale('log')
#plt.show()


