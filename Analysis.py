import torch
import numpy as np
import h5py as h5

import matplotlib.pyplot as plt

def field_statistic(confs_I, confs_C, action_I, action_C, plot_layout_dict, param, skip = 1):
    """
    :param confs_I: Configurations on the original (real/tangent) plane
    :param confs_C: Configurations on the Sign Reduced Manifold
    :param action_I: Action values of confs_I
    :param action_C: Action values of confs_C
    :param plot_layout_dict: {'title': str, 'I str': str, 'C str': str}
    :param param: isle.util.parameters
    :param skip: how many confs should be skipped in action plots etc.
    :return: fig_phys,fig_stat to save
    """

    # define FZJ cooperate colors
    C1 = (  2/255, 61/255,107/255)
    C2 = (175/255,130/255,185/255)

    fig_phys,axs_phys = plt.subplots(1,2,figsize=(18,4.5))

    fig_stat = plt.figure(figsize=(18,12))#.subplots(2,2,figsize=(18,8))

    # further functions require output to be complex valued thus add a 0
    if not confs_I.is_complex():
        confs_I = confs_I.to(torch.cdouble)
    if not confs_C.is_complex():
        confs_C = confs_C.to(torch.cdouble)

    def phys():
        # action evolution
        # we need to reduce the amount (taking every 50th only) to increase readability of the plot

        fig_phys.suptitle(f"Action Statistics: {plot_layout_dict['title']}" )

        axs_phys[0].plot(np.arange(action_I.size(0))[::skip],action_I.real[::skip],'.',color=C1,label=f"{plot_layout_dict['I str']} $Re(S)$",alpha=0.3)
        axs_phys[0].plot(np.arange(action_I.size(0))[::skip],action_I.imag[::skip],'+',color=C1,label=f"{plot_layout_dict['I str']} $Im(S)$",alpha=0.3)
        axs_phys[0].plot(np.arange(action_C.size(0))[::skip],action_C.real[::skip],'.',color=C2,label=f"{plot_layout_dict['C str']} $Re(S)$",alpha=0.3)
        axs_phys[0].plot(np.arange(action_C.size(0))[::skip],action_C.imag[::skip],'+',color=C2,label=f"{plot_layout_dict['C str']} $Im(S)$",alpha=0.3)

        axs_phys[0].plot(np.arange(action_I.size(0))[::skip],
                         np.ones(action_I.size(0)//skip)*np.mean(action_I.real.numpy()),
                         '-',
                         color=C1,
                         label=rf"{plot_layout_dict['I str']} $\langle Re(S)\rangle$"
                         )
        axs_phys[0].plot(np.arange(action_I.size(0))[::skip],
                         np.ones(action_I.size(0)//skip)*np.mean(action_I.imag.numpy()),
                         '--',
                         color=C1,
                         label=rf"{plot_layout_dict['I str']} $\langle Im(S)\rangle$"
                         )
        axs_phys[0].plot(np.arange(action_C.size(0))[::skip],
                         np.ones(action_C.size(0)//skip)*np.mean(action_C.real.numpy()),
                         '-',
                         color=C2,
                         label=rf"{plot_layout_dict['C str']} $\langle Re(S)\rangle$"
                         )
        axs_phys[0].plot(np.arange(action_C.size(0))[::skip],
                         np.ones(action_C.size(0)//skip)*np.mean(action_C.imag.numpy()),
                         '--',
                         color=C2,
                         label=rf"{plot_layout_dict['C str']} $\langle Im(S)\rangle$"
                         )

        axs_phys[0].set_title("Action Evolution")
        axs_phys[0].legend(
            bbox_to_anchor=(-0.5,1),
            loc='upper left'
            #     fontsize='xx-small'
        )
        axs_phys[0].set_xlabel("Markov Chain ID")
        axs_phys[0].set_ylabel("S")


        # Im[S] histogram
        axs_phys[1].hist(
            x = action_I.imag.numpy(),
            bins = 70,
            density = True,
            color=C1,
            label = f"{plot_layout_dict['I str']}"
        )
        axs_phys[1].hist(
            x = action_C.imag.numpy(),
            bins = 70,
            density = True,
            color=C2,
            label = f"{plot_layout_dict['C str']}",
            alpha=0.7
        )
        axs_phys[1].set_title(r"$Im(S)$ Distribution")

        axs_phys[1].legend(
            bbox_to_anchor=(1.05,1),
            loc='upper left'
            #     fontsize='xx-small'
        )
        axs_phys[1].set_xlabel(r"Im(S)")
        axs_phys[1].set_ylabel("Density")


    def stat():
        Nt = param['nt']
        Nx = param['volume']//Nt


        def fieldDiagnostics(phi):

            mean = torch.mean(phi,dim=-1)
            norm_r = torch.linalg.norm(phi.real,dim=-1)
            norm_i = torch.linalg.norm(phi.imag,dim=-1)

            return mean,norm_r,norm_i


        def modPi(S):
            if np.isnan(S).any():
                print("Warning: Found S = nan; returning 0 instead")
                S[np.isnan(S)] = 0
            return (S + np.pi) % (2 * np.pi) - np.pi


        def plot_fieldstats():
            Opt_I = dict(ls="", marker=".", c=C1, label=plot_layout_dict['I str'])
            Opt_C = dict(ls="", marker=".", c=C2, alpha=0.7, label=plot_layout_dict['C str'])

            # === Action reports ===
            # Re[S](Re[<Phi>])
            axs = fig_stat.add_subplot(3, 4, 1)
            axs.set_xlabel(r"Re $\langle\phi\rangle$")
            axs.set_ylabel(r"Re $S$")
            axs.plot(meanPhi_I.real, action_I.real, **Opt_I)
            axs.plot(meanPhi_C.real, action_C.real, **Opt_C)

            # Re[S](Im[<Phi>])
            axs = fig_stat.add_subplot(3, 4, 2)
            axs.set_xlabel(r"Im $\langle\phi\rangle$")
            axs.set_ylabel(r"Re $S$")
            axs.plot(meanPhi_I.imag, action_I.real, **Opt_I)
            axs.plot(meanPhi_C.imag, action_C.real, **Opt_C)

            # Im[S](Re[<Phi>])
            axs = fig_stat.add_subplot(3, 4, 3)
            axs.set_xlabel(r"Re $\langle\phi\rangle$")
            axs.set_ylabel(r"Im $S$ mod $2\pi$")
            axs.plot(meanPhi_I.real, modPi(action_I.imag), **Opt_I)
            axs.plot(meanPhi_C.real, modPi(action_C.imag), **Opt_C)

            # Im[S](Im[<Phi>])
            axs = fig_stat.add_subplot(3, 4, 4)
            axs.set_xlabel(r"Im $\langle\phi\rangle$")
            axs.set_ylabel(r"Im $S$ mod $2\pi$")
            axs.plot(meanPhi_I.imag, modPi(action_I.imag), **Opt_I)
            axs.plot(meanPhi_C.imag, modPi(action_C.imag), **Opt_C)

            # Re[S](Re |phi|)
            axs = fig_stat.add_subplot(3, 4, 5)
            axs.set_xlabel(r"Re |$\phi$|")
            axs.set_ylabel(r"Re $S$")
            axs.plot(normRealPhi_I, action_I.real, **Opt_I)
            axs.plot( normRealPhi_C, action_C.real, **Opt_C)

            # Re[S](Im |phi|)
            axs = fig_stat.add_subplot(3, 4, 6)
            axs.set_xlabel(r"Im |$\phi$|")
            axs.set_ylabel(r"Re $S$")
            axs.plot(normRealPhi_I, action_I.real, **Opt_I)
            axs.plot( normRealPhi_C, action_C.real, **Opt_C)

            # Im[S](Re |phi|)
            axs = fig_stat.add_subplot(3, 4, 7)
            axs.set_xlabel(r"Re |$\phi$|")
            axs.set_ylabel(r"Im $S$ mod $2\pi$")
            axs.plot(normRealPhi_I, modPi(action_I.imag), **Opt_I)
            axs.plot( normRealPhi_C, modPi(action_C.imag), **Opt_C)

            # Im[S](Im |phi|)
            axs = fig_stat.add_subplot(3, 4, 8)
            axs.set_xlabel(r"Im |$\phi$|")
            axs.set_ylabel(r"Im $S$ mod $2\pi$")
            axs.plot(normRealPhi_I, modPi(action_I.imag), **Opt_I)
            axs.plot( normRealPhi_C, modPi(action_C.imag), **Opt_C)

            # === Bare field reports ===
            # Im[<Phi>](Re[<Phi>])
            axs = fig_stat.add_subplot(3, 4, 9)
            axs.set_xlabel(r"Re $\langle\phi\rangle$")
            axs.set_ylabel(r"Im $\langle\phi\rangle$")
            axs.plot(meanPhi_I.real, meanPhi_I.imag, **Opt_I)
            axs.plot(meanPhi_C.real, meanPhi_C.imag, **Opt_C)

            # |Im(phi)|(|Re(phi)|)
            axs = fig_stat.add_subplot(3, 4, 10)
            axs.set_xlabel(r"|$\mathrm{Re}\phi$|")
            axs.set_ylabel(r"|$\mathrm{Im}\phi$|")
            axs.plot(normRealPhi_I, normImagPhi_I, **Opt_I)
            axs.plot( normRealPhi_C, normImagPhi_C, **Opt_C)

            # Im[Phi](Re[Phi])
            axs = fig_stat.add_subplot(3, 4, 11)
            axs.set_xlabel(r"Re $\phi$")
            axs.set_ylabel(r"Im $\phi$")
            axs.plot(confs_I.real.flatten(), confs_I.imag.flatten(), **Opt_I)
            axs.plot(confs_C.real.flatten(), confs_C.imag.flatten(), **Opt_C)

            handles, labels = axs.get_legend_handles_labels()
            axs = fig_stat.add_subplot(3, 4, 12)
            axs.axis('off')
            fig_stat.suptitle(f"Field Statistics: {plot_layout_dict['title']}" )
            axs.legend(handles, labels, loc='center',
                       ncol=1, fancybox=True, shadow=True,
                       prop={'size': 18},
                       markerscale=4,
                       )

        meanPhi_I, normRealPhi_I, normImagPhi_I = fieldDiagnostics(confs_I)
        meanPhi_C, normRealPhi_C, normImagPhi_C = fieldDiagnostics(confs_C)

        plot_fieldstats()


    phys()
    stat()

    fig_phys.tight_layout()
    fig_stat.tight_layout()

    return fig_phys,fig_stat


def estimate_correlators(fn,analysisParam_list, Nconf_sizes):
    from qcdanalysistools import dataAnalysis


    # read in the simulation data
    with h5.File(fn,'r') as h5f:
        actVals = h5f["weights/actVal"][()]
        try:
            logDetJ = h5f["weights/logdetJ"][()]
        except(KeyError):
            logDetJ = np.zeros_like(actVals)
        numerCorr_cd = h5f["correlation_functions/single_particle/creation_destruction"][()]
        numerCorr_dc = h5f["correlation_functions/single_particle/destruction_creation"][()]

    # prepare some named lists for later use
    corrEst_cd = np.zeros(shape=(len(Nconf_sizes),*numerCorr_cd.shape[1:]),dtype=complex)
    corrVar_cd = np.zeros(shape=(len(Nconf_sizes),*numerCorr_cd.shape[1:]))
    corrEst_dc = np.zeros(shape=(len(Nconf_sizes),*numerCorr_cd.shape[1:]),dtype=complex)
    corrVar_dc = np.zeros(shape=(len(Nconf_sizes),*numerCorr_cd.shape[1:]))
    ImSeffWeightsEst = np.zeros(shape=(len(Nconf_sizes)),dtype=complex)
    ImSeffWeightsVar = np.zeros(shape=(len(Nconf_sizes)))

    # compute Seff
    Seff = actVals - logDetJ

    # get e^(i Im[Seff]) used for reweighting
    ImSeffWeights = np.exp(-1j*Seff.imag)

    for size_id in range(len(Nconf_sizes)):
        # compute the estimate and variance for <e^(-i Im[Seff])>
        ImSeffWeightsEst[size_id], ImSeffWeightsVar[size_id] = dataAnalysis(analysisParam_list[size_id],
                                                                            ImSeffWeights[:Nconf_sizes[size_id]])

        # compute the estimate and variance for < C_cd * e^(-i Im[Seff]) >
        corrEst_cd[size_id], corrVar_cd[size_id] = dataAnalysis(analysisParam_list[size_id],
                                                                numerCorr_cd[:Nconf_sizes[size_id]]*ImSeffWeights[:Nconf_sizes[size_id],np.newaxis,np.newaxis,np.newaxis]
                                                                )

        # compute the file estimate < C_cd * e^(- i Im[Seff]) > / <e^(-i Im[Seff])>
        corrEst_cd[size_id]/= ImSeffWeightsEst[size_id]

        # compute the estimate and variance for < C_dc * e^(- i Im[Seff]) >
        corrEst_dc[size_id], corrVar_dc[size_id] = dataAnalysis(analysisParam_list[size_id],
                                                                numerCorr_dc[:Nconf_sizes[size_id]]*ImSeffWeights[:Nconf_sizes[size_id],np.newaxis,np.newaxis,np.newaxis]
                                                                )

        # compute the file estimate < C_dc *e^(- i Im[Seff]) > / <e^(-i Im[Seff])>
        corrEst_dc[size_id]/= ImSeffWeightsEst[size_id]

    return {
        "Corr cd": {"Est": corrEst_cd, "Var": corrVar_cd},
        "Corr dc": {"Est": corrEst_dc, "Var": corrVar_dc},
        "Weights": {"Est": ImSeffWeightsEst, "Var": ImSeffWeightsVar}
    }