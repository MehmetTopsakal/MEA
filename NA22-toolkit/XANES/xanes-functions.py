# XANES 
import os, sys, datetime
import glob, linecache, shutil
from larch.io import read_ascii, read_athena
from larch.xafs import find_e0, pre_edge, autobk, xftf
from larch import Group
import larch
from copy import deepcopy
from datatree import DataTree

# General
import numpy as np
import xarray as xr

# Plotting 
import matplotlib.pyplot as plt

# X-ray database
import xraydb as xdb

########### Read files ##########
# reads scan files in a directory and sorts by experiment start time
def read_files(
    pattern = None,
    fl_in = None,
    mode=["ISS",56],
    exclude_these=[],
    labels_str=None,
    plot_mus=True,
    plot_channels=False,
    plot_filenames=True,
    sdd=False,
    sdd_cols=[9, 9 + 4, 9 + 4 + 4, 9 + 4 + 4 + 4],
    return_dt = True,
    verbose = False,
):

    reads = []

    if fl_in is not None:
        fl = fl_in
        pattern = ''
    else:
        fl = sorted(glob.glob(pattern))

    for e, f in enumerate(fl):
        if verbose:
            print('reading %s'%(f))
        try:
            # read file
            d = np.loadtxt(f, unpack=True)

            if mode[0] == "ISS":
                l = linecache.getline(f, mode[1])
                dt = datetime.datetime.strptime(
                    "%s_%s" % (l.split()[2], l.split()[3][:8]), "%m/%d/%Y_%H:%M:%S"
                )
            if mode[0] == "ISS_old":
                l = linecache.getline(f, mode[1])
                dt = datetime.datetime.strptime(
                    "%s_%s" % (l.split()[3], l.split()[4][:8]), "%m/%d/%Y_%H:%M:%S"
                )
            elif mode[0] == "ISS_2021_3":
                l = linecache.getline(f, mode[1])
                dt = datetime.datetime.strptime(
                    "%s_%s" % (l.split()[2], l.split()[3][:8]), "%m/%d/%Y_%H:%M:%S"
                )
            elif mode[0] == "QAS":
                l = linecache.getline(f, mode[1])
                dt = datetime.datetime.strptime(
                    "%s_%s" % (l.split()[3], l.split()[4]), "%m/%d/%Y_%H:%M:%S"
                )
            elif mode[0] == "BMM":
                l = linecache.getline(f, mode[1])
                dt = datetime.datetime.strptime(
                    l, "# Scan.start_time: %Y-%m-%dT%H:%M:%S\n"
                )
            elif mode[0] == "12BM":
                l = linecache.getline(f, mode[1])
                dt = datetime.datetime.strptime(l, "#D %a %b %d %H:%M:%S %Y \n")
            elif mode[0] == "20ID":
                l = linecache.getline(f, mode[1]).split()
                dt = datetime.datetime.strptime(
                    "%s_%s_%s" % (l[9], l[10], l[11][0:2]), "%m/%d/%Y_%I:%M:%S_%p"
                )
            elif mode[0] == "SRX":
                l = linecache.getline(f, mode[1])
                dt = datetime.datetime.strptime("%s_%s_%s_%s" % (l.split()[3], l.split()[4], l.split()[5], l.split()[6]), "%b_%d_%H:%M:%S_%Y")                
            reads.append([dt.timestamp(), dt.isoformat(), f, d])

        except Exception as exc:
            if verbose:
                print(exc)
                print("Unable to read %s" % (f))
            else:
                pass

    # sort by timestamp
    reads.sort(key=lambda x: x[0])

    reads = [i for j, i in enumerate(reads) if j not in exclude_these]
    

    if labels_str is None:
        # figure out columns label from first file
        f0 = open(reads[0][2], "r")
        for e, line in enumerate(f0):
            if line.startswith("#"):
                last_comment_line = e
        col_labels_line = linecache.getline(reads[0][2], last_comment_line + 1).replace(
            "#", ""
        )
        col_labels = col_labels_line.split()

    else:
        col_labels = labels_str.replace("#", "").split()
        
        
    if mode[0] == 'SRX':
        col_labels = ['energy', 'energy_bragg', 'energy_c2_x', 'sclr_im', 'i0', 'it']
        
    if mode[0] == 'SRX':    
        col_energy = col_labels.index("energy")
        col_i0 = col_labels.index("i0")
        col_it = col_labels.index("it")
        col_ir = None
        col_if = None
    else: 
        col_energy = col_labels.index("energy")
        col_i0 = col_labels.index("i0")
        col_it = col_labels.index("it")
        col_ir = col_labels.index("ir")
        col_if = col_labels.index("iff")  

    if plot_channels:
        
        
        if mode[0] == 'SRX': 
            fig = plt.figure(figsize=(4,4))
            ax = fig.add_subplot(1, 1, 1)
            for er,i in enumerate(reads):
                for e, c in enumerate(sdd_cols):
                    if er < 1:
                        ax.plot(1000*i[3][col_energy], i[3][c], color="C%s" % e,label='%d'%e)
                    else:
                        ax.plot(1000*i[3][col_energy], i[3][c], color="C%s" % e)

            plt.legend()
            plt.tight_layout()
            
        else:

            mosaic = """
                    ABE
                    CDE
                    """
            fig = plt.figure(figsize=(8, 6),layout="constrained",dpi=128)
            axes = fig.subplot_mosaic(mosaic)

            # ax = fig.add_subplot(2, 4, 1)
            ax = axes["A"]
            for i in reads:
                if verbose:
                    print("%s [%d,%.2f,%.2f]" % (i[2].split('/')[-1][-30:], e, i[3][0][0], i[3][0][-1]))
                ax.plot(i[3][col_energy], i[3][col_i0])
            ax.set_ylabel("I$_0$")

            # ax = fig.add_subplot(2, 4, 2)
            ax = axes["B"]
            for i in reads:
                ax.plot(i[3][col_energy], i[3][col_it])
            ax.set_ylabel("I$_t$")
            
            if col_ir is not None:
                ax = axes["C"]
                for i in reads:
                    ax.plot(i[3][col_energy], i[3][col_ir])
                ax.set_xlabel("Energy, eV")
                ax.set_ylabel("I$_r$")


            if col_if is not None:
                ax = axes["D"]
                for i in reads:
                    ax.plot(i[3][col_energy], i[3][col_if])
                ax.set_ylabel("I$_{ff}$")
                ax.set_xlabel("Energy, eV")


            ax = axes["E"]

            ax.axis("off")
            ax.set_title(pattern.split('/')[-1],fontsize=10)
            dy = 1 / len(reads)
            if plot_filenames:
                for e, i in enumerate(reads):
                    ax.text(
                        -0.3,
                        e * dy,
                        "%s [%d,%.2f,%.2f]" % (i[2].split('/')[-1][-30:], e, i[3][0][0], i[3][0][-1]),
                        color="C%d" % (e % 10),
                        transform=ax.transAxes,
                        fontsize=6,
                    )

            plt.tight_layout()

            if sdd:
                fig = plt.figure(figsize=(4,4))
                ax = fig.add_subplot(1, 1, 1)
                for i in reads:
                    for e, c in enumerate(sdd_cols):
                        ax.plot(i[3][col_energy], i[3][c], color="C%s" % e)

                plt.tight_layout()

    if plot_mus:
        fig = plt.figure(figsize=(10,4),dpi=128)

        if mode[0] == 'SRX':
            ax_t = fig.add_subplot(1, 3, 1)
            ax_f = fig.add_subplot(1, 3, 2)
            if plot_filenames:
                ax_l = fig.add_subplot(1, 3, 3)
            f_sign = 1
            escale = 1000
        else:
            ax_t = fig.add_subplot(1, 4, 1)
            ax_r = fig.add_subplot(1, 4, 2)
            ax_f = fig.add_subplot(1, 4, 3)
            if plot_filenames:
                ax_l = fig.add_subplot(1, 4, 4)
            f_sign = -1
            escale = 1000
                
                        
        for i in reads:
            ax_t.plot(escale*i[3][col_energy], -np.log(i[3][col_it] / i[3][col_i0]))
        # ax.set_ylabel('$\mu_{transmission}$')
        ax_t.set_xlabel("Energy, eV")
        ax_t.set_title("Transmission")

        if col_ir is not None:
            for i in reads:
                ax_r.plot(escale*i[3][col_energy], -np.log(i[3][col_ir] / i[3][col_it]))
            ax_r.set_xlabel("Energy, eV")
            ax_r.set_title("Reference")
            


        if sdd:
            for i in reads:
                ax_f.plot(escale*i[3][col_energy],(f_sign*(np.array([i[3][s] for s in sdd_cols]).mean(axis=0))/i[3][col_i0]))
        else:
            for i in reads:
                ax_f.plot(escale*i[3][col_energy], (i[3][col_if] / i[3][col_i0]))
        ax_f.set_xlabel("Energy, eV")
        # ax.set_ylabel('$\mu_{fluorescence}$')
        if sdd:
            ax_f.set_title("Fluorescence (SDD)")
        else:
            ax_f.set_title("Fluorescence")


        if plot_filenames:
            ax_l.axis("off")
            dy = 1 / len(reads)
            for e, i in enumerate(reads):
                ax_l.text(
                    -0.2,
                    e * dy,
                    "%s [%d,%.2f,%.2f]" % (i[2].split('/')[-1][-30:], e, i[3][0][0], i[3][0][-1]),
                    color="C%d" % (e % 10),
                    transform=ax_l.transAxes,
                    fontsize=6,
                )

        plt.tight_layout()

    if return_dt:
        try:
            if mode[0] == 'SRX':
                E = 1000*np.array([i[3][col_energy] for i in reads]).mean(axis=0)
            else:
                E = np.array([i[3][col_energy] for i in reads]).mean(axis=0)
                
            mus_trans = np.array([-np.log(i[3][col_it] / i[3][col_i0]) for i in reads])
            
            if mode[0] == 'SRX':
                f_sign = 1
            else:
                f_sign = -1
                mus_ref = np.array([np.log(i[3][col_ir] / i[3][col_it]) for i in reads])
                
            if sdd:
                mus_fluo = np.array([(f_sign*(np.array([i[3][s] for s in sdd_cols]).mean(axis=0))/i[3][col_i0]) for i in reads])
            else:
                mus_fluo = f_sign*np.array([(i[3][col_if] / i[3][col_i0]) for i in reads])

            ds_dict = {}
            
            if mode[0] == 'SRX':
                for d in [
                    [mus_trans, "transmission"],
                    [mus_fluo, "fluoresence"],
                ]:
                    ds = xr.Dataset()
                    ds["mus"] = xr.DataArray(
                        data=d[0], coords=[np.arange(len(reads)), E], dims=["scan_num", "energy"])
                    ds_dict[d[1]] = ds
            else:
                for d in [
                    [mus_trans, "transmission"],
                    [mus_ref, "reference"],
                    [mus_fluo, "fluoresence"],
                ]:
                    ds = xr.Dataset()
                    ds["mus"] = xr.DataArray(
                        data=d[0], coords=[np.arange(len(reads)), E], dims=["scan_num", "energy"])
                    ds_dict[d[1]] = ds   
                
                
                
                
                
                
            dt = DataTree.from_dict(ds_dict)
            dt.attrs["files"] = [i[2] for i in reads]
            dt.attrs["sdd"] = str(sdd)
            dt.attrs["mode"] = mode
            dt.attrs["pattern"] = pattern
            if sdd:
                dt.attrs["sdd_cols"] = sdd_cols

            return dt
        except Exception as exc:
            print(exc)
            print('\n Unable to get dt, something is wrong...\nreturning reads')
            return reads
    else:
        return reads


########## Process Dataset ##########
def process_ds(
    ds_in,
    e0=None,
    glitches=[],
    pre1=None,
    pre2=None,
    norm1=None,
    norm2=None,
    nvict=2,
    rbkg=1.15,
    kweight=2,
    kmin=2,
    kmax=10,
    dk=0.1,
    window="hanning",
    plot_raw=True,
    calc_xftf = True
):
    if e0 is None:
        e0 = find_e0(ds_in.energy.values, ds_in.mus.mean(axis=0).values)

    if glitches is not None or []:
        Is_new = []
        for i in ds_in.mus:
            Enew, Inew = i.energy.values.copy(), i.values.copy()
            for g in glitches:
                Etmp = [
                    Enew[e]
                    for e, s in enumerate(Enew)
                    if (s < float(g.split(":")[0]) or s > float(g.split(":")[1]))
                ]
                Itmp = [
                    Inew[e]
                    for e, s in enumerate(Enew)
                    if (s < float(g.split(":")[0]) or s > float(g.split(":")[1]))
                ]
                Enew, Inew = np.array(Etmp), np.array(Itmp)
            Is_new.append(Inew)
        Is_new = np.array(Is_new)
        da_mus = xr.DataArray(
            data=Is_new,
            coords=[np.arange(Is_new.shape[0]), Enew],
            dims=["scan_num", "energy"],
        )
    else:
        da_mus = ds_in.mus

    # pre_edge and normalization parameters
    if pre1 is None:
        pre1 = -round(e0 - da_mus.energy.values[1])
    if pre2 is None:
        pre2 = round(pre1 / 3)
    if norm2 is None:
        norm2 = round(da_mus.energy.values[-2] - e0)
    if norm1 is None:
        norm1 = round(norm2 / 3)

    flats = []
    for d in da_mus:
        group = Group(energy=da_mus.energy.values, mu=d.values, filename=None)
        pre_edge(
            group,
            e0=e0,
            pre1=pre1,
            pre2=pre2,
            norm1=norm1,
            norm2=norm2,
            nvict=nvict,
            group=group,
        )
        flats.append(group.flat)
    flats = np.array(flats)
    da_flats = xr.DataArray(
        data=flats,
        coords=[np.arange(da_mus.shape[0]), da_mus.energy.values],
        dims=["scan_num", "energy"],
    )



    group = Group(
        energy=da_flats.energy.values, mu=da_flats.mean(axis=0).values, filename=None
    )
    pre_edge(
        group,
        e0=e0,
        pre1=pre1,
        pre2=pre2,
        norm1=norm1,
        norm2=norm2,
        nvict=nvict,
        group=group,
    )

    ds = xr.Dataset()
    ds["mu"] = xr.DataArray(data=group.mu, coords=[group.energy], dims=["energy"])
    ds["mus"] = da_mus
    ds["norm"] = xr.DataArray(data=group.norm, coords=[group.energy], dims=["energy"])
    ds["flat"] = xr.DataArray(data=group.flat, coords=[group.energy], dims=["energy"])
    ds["dmude"] = xr.DataArray(data=group.dmude, coords=[group.energy], dims=["energy"])
    ds["pre_edge"] = xr.DataArray(
        data=group.pre_edge, coords=[group.energy], dims=["energy"]
    )
    ds["post_edge"] = xr.DataArray(
        data=group.post_edge, coords=[group.energy], dims=["energy"]
    )
    # ds.attrs['pre_edge_details'] = group.pre_edge_details.__dict__

    ds.attrs["e0"] = e0
    ds.attrs["pre1"] = pre1
    ds.attrs["pre2"] = pre2
    ds.attrs["nvict"] = nvict
    ds.attrs["norm1"] = norm1
    ds.attrs["norm2"] = norm2

    fig = plt.figure(figsize=(10, 5))
    
    
    if calc_xftf:
        ax = fig.add_subplot(1, 2, 1)
        ax_in = fig.add_axes([0.27, 0.18, 0.18, 0.35])
    else:
        if plot_raw:
            ax = fig.add_subplot(1, 2, 2)
            ax_in = fig.add_subplot(1, 2, 1)   
        else:
            ax = fig.add_subplot(1, 2, 1)
            
    
    
    ax.plot(group.energy, group.flat, color="k", lw=2)

    ax.axvline(
        group.pre_edge_details.pre1 + group.e0, linewidth=0.3, color="k", linestyle="--"
    )
    ax.axvline(
        group.pre_edge_details.pre2 + group.e0, linewidth=0.3, color="k", linestyle="--"
    )
    ax.axvline(group.e0, linewidth=0.3, color="k", linestyle="--")
    ax.axvline(
        group.pre_edge_details.norm1 + group.e0,
        linewidth=0.3,
        color="k",
        linestyle="--",
    )
    ax.axvline(
        group.pre_edge_details.norm2 + group.e0,
        linewidth=0.3,
        color="k",
        linestyle="--",
    )
    ax.set_xlim(group.pre_edge_details.pre2 + group.e0,group.pre_edge_details.norm1 + group.e0)
    ax.set_ylabel("Normalized $\mu(E)$", fontsize = 16)
    ax.set_xlabel("Energy, eV", fontsize = 16)
    ax.tick_params(axis='both', which='major', labelsize=14)

    if plot_raw:
        # ax_in.plot(group0.energy, group0.mu, color="k", lw=1)
        # ax_in.plot(group0.energy, group0.pre_edge, lw=0.5)
        # ax_in.plot(group0.energy, group0.post_edge, lw=0.5)
        for m in ds_in.mus:
            m.plot(ax=ax_in)
        ax_in.set_title(None)
        ax_in.set_xlabel(None)
        ax_in.set_ylabel(None)
        if glitches is not None or []:
            for g in glitches:
                ax_in.axvline(
                    x=float(g.split(":")[0]),
                    linewidth=0.2,
                    color="k",
                    linestyle="--",
                    alpha=0.5,
                )
                ax_in.axvline(
                    x=float(g.split(":")[1]),
                    linewidth=0.2,
                    color="k",
                    linestyle="--",
                    alpha=0.5,
                )

    try:
        if calc_xftf:
            autobk(group, rbkg=rbkg, kweight=kweight)
            xftf(group, kmin=kmin, kmax=kmax, dk=dk, kwindow=window)
            ds["bkg"] = xr.DataArray(data=group.bkg, coords=[group.energy], dims=["energy"])

            # if plot_raw:
            #     ds["bkg"].plot(ax=ax_in,color='r')
            #     ds["pre_edge"].plot(ax=ax_in,color='r')
            #     ds["post_edge"].plot(ax=ax_in,color='r')

            ds.attrs["rbkg"] = rbkg
            ds.attrs["kweight"] = kweight
            ds.attrs["kmin"] = kmin
            ds.attrs["kmax"] = kmax
            ds.attrs["dk"] = dk
            ds.attrs["window"] = window

            ds["chir_mag"] = xr.DataArray(data=group.chir_mag, coords=[group.r], dims=["r"])
            ds["chir_re"] = xr.DataArray(data=group.chir_re, coords=[group.r], dims=["r"])
            ds["chir_im"] = xr.DataArray(data=group.chir_im, coords=[group.r], dims=["r"])

            ds["kwin"] = xr.DataArray(data=group.kwin, coords=[group.k], dims=["k"])
            ds["k2chi"] = xr.DataArray(
                data=group.k * group.k * group.chi, coords=[group.k], dims=["k"]
            )

            ax = fig.add_subplot(1, 2, 2)
            ax.plot(group.r, group.chir_mag, "-b", lw=2)

            ax.set_xlim([0, 7])
            ax.set_xlabel("$\it{R}$ ($\AA$)")
            ax.set_ylabel("|$\chi$ ($\it{R}$)| ($\AA^{-3}$)")
            ax.set_title(
                "rbkg=%.2f, kmin=%.2f, kmax=%.2f \nkweight=%.2f, dk=%.2f, kwindow=%s"
                % (rbkg, kmin, kmax, kweight, dk, window),
                fontsize=9,
            )

            ax = fig.add_axes([0.77, 0.57, 0.2, 0.3])
            ax.plot(group.k, group.k * group.k * group.chi, "-r")
            ax.axvline(x=kmin, linestyle=":", color="k")
            ax.axvline(x=kmax, linestyle=":", color="k")
            ax.set_xlabel("$\it{k}$ ($\AA^{-1}$)")
            ax.set_ylabel("$\it{k^{2}}$ $\chi$ ($\it{k}$) ($\AA^{-2}$)")

    except Exception as exc:
        print(exc)
        print("Unable to get xftf")

    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    plt.tight_layout()

    return ds