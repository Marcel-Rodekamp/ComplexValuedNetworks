import argparse 
import torch
import numpy as np
import h5py as h5
import itertools
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from pathlib import Path
from time import time_ns
import isle
from Misc import read_params, makeIsleParams, get_fn, writeParamsH5, make_action, overview
from ML_Layer import LinearTransformation, AffineCoupling, Equivariance, PRCL, Sequential, Flow
from ML_Loss import Integrator, IntegrateHolomorphicFlowLoss
from ML_Training import train
from HMC import ML_HMC, HMC
from Correlators import measure_correlators, estimate_correlators, estimate_statistical_power
from Hubbard2SiteModel import Hubbard2SiteModel
from RungeKutta import singleRK4Step
# ToDo :)
# from Analysis import field_statistic

# =======================================================================
# Set up isle plotting and FZJ default colors
# =======================================================================
import isle.plotting
isle.plotting.setupMPL()

# define FZJ cooperate colors
BLUE      = (  2/255, 61/255,107/255)
LIGHTBLUE = (173/255,189/255,227/255)
GRAY      = (235/255,235/255,235/255)
RED       = (235/255, 95/255,115/255)
GREEN     = (185/255,210/255, 95/255)
YELLOW    = (250/255,235/255, 90/255)
VIOLET    = (175/255,130/255,185/255)
ORANGE    = (250/255,180/255, 90/255)
WHITE     = (255/255,255/255,255/255)

def create_model(params):
    def param_factory():
        nn = []
        for _ in range(1,params['Number Internal Layers']):
            # NxLayer = Nx/2 = 1 
            L = LinearTransformation(Nt = params['Nt'], Nx = params['Nx']//2)
            torch.nn.init.zeros_(L.weight)
            torch.nn.init.zeros_(L.bias)
        
            nn.append(L)
            nn.append(torch.nn.Softsign())
        
        L = LinearTransformation(Nt = params['Nt'], Nx = params['Nx']//2)
        torch.nn.init.zeros_(L.weight)
        torch.nn.init.zeros_(L.bias)
        
        nn.append(L)
        return torch.nn.Sequential(*nn)
    
    def PRCL_factory():
        return PRCL(
            Nt = params['Nt'], 
            Nx = params['Nx'], 
            coupling = AffineCoupling(
                m = param_factory(),
                a = param_factory()
            )
        )
    
    return Equivariance(PRCL_factory())


def generateTrainingData(params, HM):
    print("Generating RK4 train data...")
    time = time_ns() * 1e-9
    # Training Data
    trainData_RK4 = torch.randn(size = (params['Ntrain']['RK4'],params['Nt'],params['Nx']), dtype = torch.double) + 1j*params['Tangent Plane Offset']
    trainTargetData_RK4 = singleRK4Step(params['Flow Step Size'], trainData_RK4, HM)
    trainDataset_RK4 = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(trainData_RK4, trainTargetData_RK4),
        batch_size = params['Ntrain']['RK4']
    )
    
    
    # Validation Data
    validData_RK4 = torch.randn(size = (params['Nvalid']['RK4'],params['Nt'],params['Nx']), dtype = torch.double) + 1j*params['Tangent Plane Offset']
    validTargetData_RK4 = singleRK4Step(params['Flow Step Size'], validData_RK4, HM)
    validDataset_RK4 = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(validData_RK4, validTargetData_RK4),
        batch_size = params['Nvalid']['RK4']
    )

    print(f"Generating RK4 train data took : { time_ns()*1e-9 - time:.4}s")

    print("Generating FI train data...")
    time = time_ns() * 1e-9
    # Training Data
    trainData_FI = torch.randn(size = (params['Ntrain']['FI'],params['Nt'],params['Nx']), dtype = torch.double) + 1j*params['Tangent Plane Offset']
    trainDataset_FI = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(trainData_FI),
        batch_size = params['Ntrain']['FI']
    
    )
    
    # Validation Data
    validData_FI = torch.randn(size = (params['Ntrain']['FI'],params['Nt'],params['Nx']), dtype = torch.double) + 1j*params['Tangent Plane Offset']
    validDataset_FI = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(validData_FI),
        batch_size = params['Ntrain']['FI']                                         
    )

    print(f"Generating FI  train data took : { time_ns()*1e-9 - time:.4}s")

    return {
        "RK4": {
            "train": trainDataset_RK4,
            "valid": trainDataset_RK4
        },
        "FI": {
            "train": trainDataset_FI,
            "valid": trainDataset_FI
        },
    }

def generateTestingData(params, HM):
    print("Generating test data...")
    time = time_ns() * 1e-9
    # Testing data
    testData = torch.randn(size = (params['Ntest'],params['Nt'],params['Nx']), dtype = torch.double) + 1j*params['Tangent Plane Offset']
    testTargetData = singleRK4Step(params['Flow Step Size'], testData, HM)
    
    print(f"Generating test data took: { time_ns()*1e-9 - time:.4}s")

    return {
        "input" :testData,
        "target":testTargetData
    }

def plotLossValues(RK4_trainLoss,RK4_validLoss,FI_trainLoss,FI_validLoss,params):
    fig = plt.figure(figsize=(9.85,4))
    
    plt.plot(
        np.arange(0,params['Number Epochs']['RK4']), RK4_trainLoss, '.', color = BLUE,
    )
    plt.plot(
        np.arange(0,params['Number Epochs']['RK4']), RK4_validLoss, '.', color = VIOLET,
    )
    plt.vlines(params['Number Epochs']['RK4']-0.5, 
        torch.min(torch.cat( [RK4_trainLoss,RK4_validLoss,FI_trainLoss,FI_validLoss], dim=0 )).item(),
        torch.max(torch.cat( [RK4_trainLoss,RK4_validLoss,FI_trainLoss,FI_validLoss], dim=0 )).item(),
        colors = 'k',
        linestyles = 'dashdot'
    )
    plt.plot(
        params['Number Epochs']['RK4']+np.arange(0,params['Number Epochs']['FI']), FI_trainLoss, '.', color = BLUE,
    )
    plt.plot(
        params['Number Epochs']['RK4']+np.arange(0,params['Number Epochs']['FI']), FI_validLoss, '.', color = VIOLET,
    )
    
    plt.text(
        x = 0.5*(params['Number Epochs']['RK4']/(params['Number Epochs']['RK4']+params['Number Epochs']['FI'])),
        y = 0.5,
        s = "RK4",
        weight = 'bold',
        fontsize = 12,
        transform = plt.gca().transAxes
    )
    plt.text(
        x = (
            (params['Number Epochs']['RK4']+0.5*params['Number Epochs']['FI']) / 
            (params['Number Epochs']['RK4']+    params['Number Epochs']['FI'])
        ),
        y = 0.5,
        s = "FI",
        weight = 'bold',
        fontsize = 12,
        transform = plt.gca().transAxes
    )
    
    plt.yscale('log')
    plt.ylabel(r"Loss $\mathcal{L}_{\mathrm{RK4}\,/\,\mathrm{FI}}$", fontsize = 12)
    plt.xlabel("Epoch", fontsize = 12)
    
    patch_1 = mpatches.Patch(color = BLUE,  label = r"Train Loss")
    patch_2 = mpatches.Patch(color = VIOLET,label = r"Valid Loss")
    
    lgd = fig.legend(handles=[patch_1,patch_2],
        loc='upper center', bbox_to_anchor=(0.5, -0.02),
        fancybox=True, shadow=True, ncol = 3
    );
    
    plt.savefig(get_fn("Results/LossPlot",".pdf",params),
        bbox_extra_artists=(lgd,), 
        bbox_inches='tight'
    )

def plotFieldStatistics(testData,testTargetData,NN,flow,params):
    # ################################################################# #
    # ################################################################# #
    # ################################################################# #
    # This function does only work if qcdanalysistools are installed!
    # ################################################################# #
    # ################################################################# #
    # ################################################################# #
    from Analysis import field_statistic
    # Get the isle action from the parameters
    lattice = isle.LATTICES[params['lattice']]
    lattice.nt(params['Nt'])
    action = make_action(lattice, makeIsleParams(params))

    V = params['Nt']*params['Nx']

    print(f"Computing Predicion")
    # Evaluate the network
    with torch.no_grad():
        RK4Pred,RK4LogDetJ = NN(testData)
        FIPred, FILogDetJ  = flow(testData)

    print(f"Computing Actions")
    # Compute actions
    S_testData = torch.zeros(len(testData), dtype = torch.cdouble)
    S_RK4Step  = torch.zeros(len(testData), dtype = torch.cdouble)
    S_RK4Pred  = torch.zeros(len(testData), dtype = torch.cdouble)
    S_FIPred  = torch.zeros(len(testData), dtype = torch.cdouble)
    for cfgID in range(len(testData)):
        # Compute action of test data
        S_testData[cfgID] = action.eval(
            isle.CDVector(testData[cfgID].reshape(V).numpy())
        )
        # Compute action of test target data (single RK4 step)
        S_RK4Step[cfgID] = action.eval(
            isle.CDVector(testTargetData[cfgID].reshape(V).numpy())
        )
        # Compute action of the NN prediction (single trained RK4 step)
        S_RK4Pred[cfgID] = action.eval(
            isle.CDVector(RK4Pred[cfgID].reshape(V).numpy())
        )
        # Compute action of the flow prediction (N trained RK4 steps)
        S_FIPred[cfgID] = action.eval(
            isle.CDVector(FIPred[cfgID].reshape(V).numpy())
        )
        
    print(f"Plotting Field Statistis (RK4-NN)")
    RK4phys,RK4stat = field_statistic(
        confs_I=testTargetData.reshape(len(testData),V), 
        confs_C=RK4Pred.reshape(len(testData),V), 
        action_I=S_RK4Step, 
        action_C=S_RK4Pred, 
        plot_layout_dict={"I str":"RK4 Step", 
                          "C str":"Trained RK4 Step", 
                          "title":"Comparing a Single RK4 Step"
        },
        param=params
    )
    
    RK4stat.savefig(get_fn("Results/NNFieldStats",".pdf",params))

    print(f"Plotting Field Statistis (TP-Flow)")
    FIphys,FIstat = field_statistic(
        confs_I=testData.reshape(len(testData),V),
        confs_C=FIPred.reshape(len(testData),V),
        action_I=S_testData,
        action_C=S_FIPred,
        plot_layout_dict={"I str":"Tangent Plane",
                          "C str":"Trained Flow",
                          "title":"Tangent Plane to The Trained Flow"
        },
        param=params
    )
    FIstat.savefig(get_fn("Results/NNFlowFieldStats",".pdf",params))

    print(f"Plotting Field Statistis (TP-RK4)")
    TPphys,TPstat = field_statistic(
        confs_I=testData.reshape(len(testData),V), 
        confs_C=testTargetData.reshape(len(testData),V), 
        action_I=S_testData, 
        action_C=S_RK4Step, 
        plot_layout_dict={"I str":"Tangent Plane", 
                          "C str":"RK4 Step", 
                          "title":"Tangent Plane to Single RK4 Step"
        }, 
        param=params
    )
    TPstat.savefig(get_fn("Results/TestDataFieldStats",".pdf",params))

def main():
    # Define a simple argument parser to handle command line arguments
    parser = argparse.ArgumentParser()

    # positional (non optional)
    parser.add_argument(
        "parameterFile",
        help = "Specify the path to the used parameter file (can be relative)."
    )
    parser.add_argument(
        "--noTraining", 
        help="Turn off the training, a model must be present in the Results/ folder to be read in!",
        action="store_true"
    )
    parser.add_argument(
        "--noTrainingData", 
        help="Turn off the training data generation! (Requires --noFieldStatistics)",
        action="store_true"
    )
    parser.add_argument(
        "--noFieldStatistics", 
        help="Turn off plotting the field statistics!",
        action="store_true"
    )
    parser.add_argument(
        "--HMCOverview", 
        help="Plot the HMC overview (typically this needs to be done only once if at all)",
        action="store_true"
    )
    parser.add_argument(
        "--HMC", 
        help="Perform HMC. Requires --MDsteps, --MDtrajLength, --NTraj, --bTherm, --sTherm to be set",
        action="store_true"
    )
    parser.add_argument(
        "--MLHMC", 
        help="Perform MLHMC with the trained/read network. Requires --MDsteps, --MDtrajLength, --NTraj, --bTherm, --sTherm to be set",
        action="store_true"
    )
    parser.add_argument(
        "--MDsteps",
        help = "Number of molecular dynamics steps"
    )
    parser.add_argument(
        "--MDtrajLength",
        help = "Length of molecular dynamics trajectory"
    )
    parser.add_argument(
        "--NTraj",
        help = "Number of trajectories (Nconf = NTraj/STherm)"
    )
    parser.add_argument(
        "--bTherm",
        help = "Number of burn in configurations"
    )
    parser.add_argument(
        "--sTherm",
        help = "Number of thermalization configurations"
    )
    parser.add_argument(
        "--MLHMCmeasureCorrs",
        help = "Turn on MLHMC correlator measurement",
        action="store_true"
    )
    parser.add_argument(
        "--HMCmeasureCorrs",
        help = "Turn on HMC correlator measurement",
        action="store_true"
    )
    parser.add_argument(
        "--plotCorrs",
        help = "Turn on plot/estimation for correlators. You should at least have 1000 measurements",
        action="store_true"
    )
    # flags to turn on various parts of the code
    args = parser.parse_args()

    # ===================================================================
    # Read Parameter File
    # ===================================================================
    paramsFN = Path(args.parameterFile).resolve()
    params = read_params(paramsFN)

    print(f"Using parameter file {paramsFN} ")
    for key in params.keys():
        print(f"{key: <22}: {params[key]}")

    # ===================================================================
    # Define the 2 Site Model
    # ===================================================================
    HM = Hubbard2SiteModel(
        Nt = params['Nt'],
        beta = params['beta'],
        U = params['U'],
        mu = params['mu'],
        tangentPlaneOffset = params['Tangent Plane Offset'] 
    )

    # ===================================================================
    # Generate Training Data
    # ===================================================================
    if not args.noTrainingData:
        print("Generating Training Data")
        trainingDataSets = generateTrainingData(params,HM)
    else:
        trainingDataSets = dict()
    
    # ===================================================================
    # Generate Testing Data
    # ===================================================================
    if not args.noFieldStatistics:
        trainingDataSets["test"] = generateTestingData(params,HM)

    # ===================================================================
    # Create the Model
    # ===================================================================
    NN = create_model(params)

    # ===================================================================
    # Prepare the Flow
    # ===================================================================
    flow = Flow(NN, params['Number Flow Steps'])
    # This is set by default. We should anyway make sure that the flow is in training mode as it has different behaviour 
    # as in the evaluation mode
    flow.train(True)
    print(flow)

    if not args.noTraining:
        # ===============================================================
        # Prepare Loss Function
        # ===============================================================
        RK4_LossFct = torch.nn.L1Loss(reduction='mean')

        FI_LossFct = IntegrateHolomorphicFlowLoss(
            Integrator(params['Flow Step Size'], HM)
        )

        # ===============================================================
        # Prepare optimizer
        # ===============================================================
        RK4_optimizer = torch.optim.Adam(NN.parameters(),lr = params['Learning Rate']['RK4'])
        FI_optimizer = torch.optim.Adam(flow.parameters(),lr = params['Learning Rate']['FI'])

        print(f"RK4 uses:\n    * {RK4_LossFct}\n    * {RK4_optimizer}")
        print(f"FI  uses:\n    * {FI_LossFct} \n    * {FI_optimizer}")


        # ===============================================================
        # RK4 Training Phase
        # ===============================================================
        RK4_trainLoss, RK4_validLoss = train(
            phaseStr  = "RK4",
            trainData = trainingDataSets['RK4']['train'], 
            validData = trainingDataSets['RK4']['valid'], 
            model     = NN, 
            optimizer = RK4_optimizer, 
            lossFct   = RK4_LossFct, 
            params    = params
        )

        # ===============================================================
        # FI Training Phase
        # ===============================================================
        FI_trainLoss, FI_validLoss = train(
            phaseStr  = "FI",
            trainData = trainingDataSets['FI']['train'], 
            validData = trainingDataSets['FI']['valid'], 
            model     = flow, 
            optimizer = FI_optimizer, 
            lossFct   = FI_LossFct, 
            params    = params
        )


        # ===============================================================
        # Plot Loss Values & Store the data to fiel
        # ===============================================================
        plotLossValues(RK4_trainLoss,RK4_validLoss,FI_trainLoss,FI_validLoss,params)
        with h5.File(get_fn("Results/Losses",".h5",params), 'w') as loss_file:
            loss_file.create_dataset(
                "RK4/train",
                data=RK4_trainLoss.numpy()
            )
            loss_file.create_dataset(
                "RK4/valid",
                data=RK4_validLoss.numpy()
            ) 
            loss_file.create_dataset(
                "FI/train",
                data=FI_trainLoss.numpy()
            )
            loss_file.create_dataset(
                "FI/valid",
                data=FI_validLoss.numpy()
            )
        
            writeParamsH5(loss_file,params)

        # ===============================================================
        # Save the model
        # ===============================================================
        torch.save({
                "NN state":NN.state_dict(),
                "RK4 optimizer state": RK4_optimizer.state_dict(),
                "FI optimizer state": FI_optimizer.state_dict(),
                "params": params
            },
            get_fn("Results/ModelCheckpoint",".pt",params)
        )
    

    else: # --noTraining == True
        # ===============================================================
        # Read the model
        # ===============================================================
        print("Reading Model")
        checkpoint = torch.load(get_fn("Results/ModelCheckpoint",".pt",params))
        NN.load_state_dict(checkpoint['NN state'])

    # ===================================================================
    # Set Flow in Inference Mode 
    # ===================================================================
    flow.eval()

    if not args.noFieldStatistics:
        plotFieldStatistics(
            testData       = trainingDataSets['test']['input'],
            testTargetData = trainingDataSets['test']['target'],
            NN = NN,
            flow = flow, 
            params= params
        )
    
    # ===================================================================
    # HMC
    # ===================================================================
    if args.HMC:
        HMC_params = params.copy()

        # Require MD steps
        if args.MDsteps is not None:
            HMC_params['MD_steps'] = int(args.MDsteps)
        else:
            raise RuntimeError("MLHMC requires argument --MDsteps")

        # Require trajectory length
        if args.MDtrajLength is not None:
            HMC_params['MD_trajectory_length'] = float(args.MDtrajLength)
        else:
            raise RuntimeError("MLHMC requires argument --MDtrajLength")

        # require number of trajectories
        if args.NTraj is not None:
            HMC_params['nTraj'] = int(args.NTraj)
        else:
            raise RuntimeError("MLHMC requires argument --NTraj")

        # require number of burn in trajectories
        if args.bTherm is not None:
            HMC_params['bTherm'] = int(args.bTherm)
        else:
            raise RuntimeError("MLHMC requires argument --bTherm")

        # require number of thermalizing in trajectories
        if args.sTherm is not None:
            HMC_params['sTherm'] = int(args.sTherm)
        else:
            raise RuntimeError("MLHMC requires argument --sTherm")

        # get file name and remove file if it exists
        f = Path("Results/HMC_Simulation.h5")
        if f.exists():
            f.unlink()

        # actually perform HMC
        HMC(f,makeIsleParams(HMC_params))

        # Store the parameters
        with h5.File(f, 'a') as h5f:
            writeParamsH5(h5f,HMC_params)

        # generate the overview
        print("Plotting overview")
        overview(f).savefig("Results/HMCSimulationOverview.pdf")

    # ===================================================================
    # ML HMC
    # ===================================================================
    if args.MLHMC:
        MLHMC_params = params.copy()

        # Require MD steps
        if args.MDsteps is not None:
            MLHMC_params['MD_steps'] = int(args.MDsteps)
        else:
            raise RuntimeError("MLHMC requires argument --MDsteps")

        # Require trajectory length
        if args.MDtrajLength is not None:
            MLHMC_params['MD_trajectory_length'] = float(args.MDtrajLength)
        else:
            raise RuntimeError("MLHMC requires argument --MDtrajLength")

        # require number of trajectories
        if args.NTraj is not None:
            MLHMC_params['nTraj'] = int(args.NTraj)
        else:
            raise RuntimeError("MLHMC requires argument --NTraj")

        # require number of burn in trajectories
        if args.bTherm is not None:
            MLHMC_params['bTherm'] = int(args.bTherm)
        else:
            raise RuntimeError("MLHMC requires argument --bTherm")

        # require number of thermalizing in trajectories
        if args.sTherm is not None:
            MLHMC_params['sTherm'] = int(args.sTherm)
        else:
            raise RuntimeError("MLHMC requires argument --sTherm")

        # get file name and remove file if it exists
        f = get_fn("Results/MLHMC_Simulation",".h5",params)
        if f.exists():
            f.unlink()

        # actually perform MLHMC
        ML_HMC(flow,f,makeIsleParams(MLHMC_params))

        # Store the parameters
        with h5.File(f, 'a') as h5f:
            writeParamsH5(h5f,MLHMC_params)

        # generate the overview
        print("Plotting overview")
        overview(f).savefig(get_fn("Results/MLHMCSimulationOverview",".pdf",params))

    # ===================================================================
    # Measure Correlators
    # ===================================================================
    if args.MLHMCmeasureCorrs:
        measure_correlators(get_fn("Results/MLHMC_Simulation",".h5",params),params)
    if args.HMCmeasureCorrs:
        measure_correlators("Results/HMC_Simulation.h5",params)

    # ===================================================================
    # Estimate and plot Correlators
    # ===================================================================
    if args.plotCorrs:
        # ================================================================
        # Estimate
        # ================================================================
        
        print("Estimating HMC Corrs with Nconf=1000 ")
        HMC_res1000 = estimate_correlators(fn = "Results/2Site_HMC_Simulation.h5", Nconf = 1_000, Nbst = 100)
        print("Estimating HMC Corrs with Nconf=10_000 ")
        HMC_res10000 = estimate_correlators(fn = "Results/2Site_HMC_Simulation.h5", Nconf = 10_000, Nbst = 100)
        print("Estimating HMC Corrs with Nconf=100_000 ")
        HMC_res100000 = estimate_correlators(fn = "Results/2Site_HMC_Simulation.h5", Nconf = 100_000, Nbst = 100)
        
        print(f"Stat Power HMC 1_000  : {HMC_res1000['StatPower'][0]:.2e} +/- {HMC_res1000['StatPower'][1]:.2e}")
        print(f"Stat Power HMC 10_000 : {HMC_res10000['StatPower'][0]:.2e} +/- {HMC_res10000['StatPower'][1]:.2e}")
        print(f"Stat Power HMC 100_000: {HMC_res100000['StatPower'][0]:.2e} +/- {HMC_res100000['StatPower'][1]:.2e}")

        with h5.File(get_fn("Results/MLHMC_Simulation",".h5",params),'r') as h5f:
            maxNconf = len(h5f['weights/actVal'])

        if(maxNconf >= 1000):
            print("Estimating MLHMC Corrs with Nconf=1000 ")
            MLHMC_res1000 = estimate_correlators(fn = get_fn("Results/MLHMC_Simulation",".h5",params), Nconf = 1_000, Nbst = 100)
        else:
            raise RuntimeError(f"Number of Correlator measurements must be at least 1000 but is {maxNconf}")

        if maxNconf >= 10_000:
            print("Estimating MLHMC Corrs with Nconf=10_000 ")
            MLHMC_res10000 = estimate_correlators(fn = get_fn("Results/MLHMC_Simulation",".h5",params), Nconf = 10_000, Nbst = 100)

        if maxNconf >= 100_000:
            print("Estimating MLHMC Corrs with Nconf=100_000 ")
            MLHMC_res100000 = estimate_correlators(fn = get_fn("Results/MLHMC_Simulation",".h5",params), Nconf = 100_000, Nbst = 100)

            
        print(f"Stat Power MLHMC 1_000  : {MLHMC_res1000['StatPower'][0]:.2e} +/- {MLHMC_res1000['StatPower'][1]:.2e}")
        if maxNconf >= 10_000:
            print(f"Stat Power MLHMC 10_000 : {MLHMC_res10000['StatPower'][0]:.2e} +/- {MLHMC_res10000['StatPower'][1]:.2e}")
        if maxNconf >= 100_000:
            print(f"Stat Power MLHMC 100_000: {MLHMC_res100000['StatPower'][0]:.2e} +/- {MLHMC_res100000['StatPower'][1]:.2e}")


        # read exact data 
        with h5.File("ExactData/exactData.h5") as h5f:
            node = f"{params['lattice']}/beta{params['beta']}/U{params['U']}/mu{params['mu']}"
            if node in h5f:
                tau = h5f["tau"][()]
                exactCorr = h5f[node][()]
                plot_exact = True 
            else:
                plot_exact = False

        # ================================================================
        # Plot
        # ================================================================
        if maxNconf >= 100_000:
            fig, axs = plt.subplots(1,3, figsize = (16,4),sharey = True)
        if maxNconf >= 10_000:
            fig, axs = plt.subplots(1,2, figsize = (16,4),sharey = True)
        else:
            fig, axs = plt.subplots(1,1, figsize = (16,4),sharey = True)
            axs = [axs]
                
        simtau = np.arange(0,params['beta'],params['beta']/params['Nt'])
        
        for i in range(2):
            axs[0].errorbar(
                simtau,HMC_res1000["Corr"][0][i,:].real, yerr=HMC_res1000["Corr"][1][i,:], fmt = '.', capsize=2,color = RED,zorder=2
            )    
            axs[0].errorbar(
                simtau,MLHMC_res1000["Corr"][0][i,:].real, yerr=MLHMC_res1000["Corr"][1][i,:], fmt = '.', capsize=2,color = BLUE,zorder=3
            )
            if plot_exact:
                axs[0].plot(
                    tau, exactCorr[i], 'k-.', zorder=1
                )
            
            if maxNconf >= 10_000:
                axs[1].errorbar(
                    simtau,HMC_res10000["Corr"][0][i,:].real, yerr=HMC_res10000["Corr"][1][i,:], fmt = '.', capsize=2,color = RED,zorder=2
                )    
                axs[1].errorbar(
                    simtau,MLHMC_res10000["Corr"][0][i,:].real, yerr=MLHMC_res10000["Corr"][1][i,:], fmt = '.', capsize=2,color = BLUE,zorder=3
                )
                if plot_exact:
                    axs[1].plot(
                        tau, exactCorr[i], 'k-.',zorder=1
                    )
            
            if maxNconf >= 100_000:
                axs[2].errorbar(
                    simtau,HMC_res100000["Corr"][0][i,:].real, yerr=HMC_res100000["Corr"][1][i,:], fmt = '.', capsize=2,color = RED,zorder=2
                )
                axs[2].errorbar(
                    simtau,MLHMC_res100000["Corr"][0][i,:].real, yerr=MLHMC_res100000["Corr"][1][i,:], fmt = '.', capsize=2,color = BLUE,zorder=3
                )
                if plot_exact:
                    axs[2].plot(
                        tau, exactCorr[i], 'k-.',zorder=1
                    )
        
        patch_1 = mpatches.Patch(color = BLUE,label = r"ML HMC")
        patch_2 = mpatches.Patch(color = RED,label = r"HMC")

        if plot_exact:
            patch_3 = mlines.Line2D([], [], color = 'k', ls='--', label="Exact Diagonalization")
            lgd = fig.legend(handles=[patch_1,patch_2,patch_3],
                loc='upper center', bbox_to_anchor=(0.5, -0.02),
                fancybox=True, shadow=True, ncol = 3
            )
        else:
            lgd = fig.legend(handles=[patch_1,patch_2],
                loc='upper center', bbox_to_anchor=(0.5, -0.02),
                fancybox=True, shadow=True, ncol = 3
            )
        
        for ax in axs:
            ax.set_yscale('log')
            ax.set_xlabel(r"$\delta t$")
        
        axs[0].set_ylabel(r"$C_{sp}(t)$");
        axs[0].set_title("$N_{\mathrm{conf}} = 1000$", fontsize=14);

        if maxNconf >= 10_000:
            axs[1].set_title("$N_{\mathrm{conf}} = 10\, 000$", fontsize=14);
        if maxNconf >= 100_000:
            axs[2].set_title("$N_{\mathrm{conf}} = 100\, 000$", fontsize=14);
        
        fig.savefig(get_fn("Results/2Site_Correlators", ".pdf", params),
                  bbox_extra_artists=(lgd,), 
                  bbox_inches='tight'
        )

        # ===============================================================
        # Estimate Statistical Power
        # ===============================================================

        if maxNconf >= 100_000:
            Nconfs = np.arange(1000,maxNconf+1,1000)
        elif maxNconf >= 10_000:
            Nconfs = np.arange(1000,maxNconf+1,100)
        else:
            Nconfs = np.arange(1,maxNconf+1,100)

        StatPowers_est = np.zeros(shape = (2,len(Nconfs)))
        StatPowers_err = np.zeros(shape = (2,len(Nconfs)))
        for i,Nconf in enumerate(Nconfs):
            StatPowers_est[0,i], StatPowers_err[0,i] = estimate_statistical_power(
                fn = "Results/2Site_HMC_Simulation.h5", 
                Nconf = Nconf, 
                Nbst = 100
            )
            StatPowers_est[1,i], StatPowers_err[1,i] = estimate_statistical_power(
                fn = get_fn("Results/MLHMC_Simulation",".h5",params), 
                Nconf = Nconf, 
                Nbst = 100
            )


        # ===============================================================
        # Plot Statistical Power
        # ===============================================================

        fig = plt.figure(figsize = (16,4) )
        plt.errorbar(
            Nconfs, StatPowers_est[0], yerr = StatPowers_err[0], capsize = 2, color = RED ,zorder = 1
        )
        plt.errorbar(
            Nconfs, StatPowers_est[1], yerr = StatPowers_err[1], capsize = 2, color = BLUE ,zorder = 2
        )
        plt.yscale('log')
        plt.ylabel(r"$\left\vert \Sigma \right\vert_{N_{\mathrm{conf}}}$", fontsize = 12)
        plt.xlabel(r"$N_{\mathrm{conf}}$", fontsize = 12)
        tit = plt.title(r"Statistical Power per $N_{\mathrm{conf}}$", fontsize = 14)
        patch_1 = mpatches.Patch(color = BLUE,label = r"ML HMC")
        patch_2 = mpatches.Patch(color = RED,label = r"HMC")
        lgd = fig.legend(handles=[patch_1,patch_2],
            loc='upper center', bbox_to_anchor=(0.5, -0.02),
            fancybox=True, shadow=True, ncol = 2
        );
        
        fig.savefig(get_fn("Results/Statpower",".pdf",params),
                  bbox_extra_artists=(lgd,tit),
                  bbox_inches='tight'
        )
    

if __name__ == "__main__":

    main()
