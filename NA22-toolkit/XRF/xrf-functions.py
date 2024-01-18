# Packages necessary
# General
import h5py
import pandas as pd
import numpy as np
import xarray as xr
from itertools import compress


# Fitting
from scipy.optimize import curve_fit
from scipy.sparse.linalg import spsolve
from scipy.linalg import cholesky
from scipy import sparse
from time import perf_counter
from scipy.signal import find_peaks, peak_prominences, peak_widths


# Plotting
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as patches

# X-ray database
import xraydb as xdb




########## Handle user inputs ##########
## Function to Convert user_input into slice
# * Inputs
#     1. user_inputs must be from built-in 'input()' function
# * Outputs
#     2. slice of range inputted by user
def input_to_slice(user_input):
    try:
        # Split the input into start, stop, and step components
        parts = user_input.split(':')

        # Convert parts to integers or leave them as None if not provided
        start = int(parts[0]) if parts[0] else None
        stop = int(parts[1]) if parts[1] else None
        

        # Create and return the slice
        return slice(start, stop)
    except ValueError:
        print("Invalid input. Please use the format 'start:stop'.")


########## Identify Elements ##########
## Function to find element fluorescence line that matches each peak
# * Inputs: 
#     1. elements: string of elements supplied by user consisting of those expected in the sample
#     2. peaks: list of energies corresponding to each peak (eV)
#     3. tolerance: percentage of acceptable difference between the database x-ray fluorescence energy and the supplied peak energies
# * Outputs: 
#     1. matched_fluor_lines: list of matched fluorescence lines showing the peak name, the element, the fluorescence line name, and the energy (eV)
#     2. matched_df: pandas dataframe of the matches
def identify_element_match(elements, peaks, tolerance):
    line_name_int = []
    identified_element = []
    energy_match = []
    matched_peak = []
    for element in elements:
        xray_line = xdb.xray_lines(element)
        line_names = list(xdb.xray_lines(element))
    
        for i in range(len(line_names)):
            fluor_energy = list(xray_line.values())[i][0] # output fluorscence energy of the selected element in the i-th index
    
            # find fluorscence line that matches to each peak
            for j, peak in enumerate(peaks):
                largest_value = max(peak,fluor_energy)
                peak_diff = (abs(fluor_energy - peak)/ largest_value)*100
    
                # find values within set tolerance threshold percent
                if peak_diff <= tolerance:
                    identified_element.append(element)
                    line_name_int.append(line_names[i])
                    energy_match.append(float(fluor_energy))
                    matched_peak.append(int(j+1))
    
    # element_emission_line = [item1 + '_' + item2 for item1, item2 in zip(identified_element, line_name_int)]
    
    # Output list of matched elements, the fluorescence line name, and the energy (eV)
    matched_fluor_lines = sorted([list(a) for a in zip(matched_peak, identified_element, line_name_int, energy_match)])
    
    column_names =  ["Peak #", "Element", "Emission Line", "Energy (eV)"]
    matched_df = pd.DataFrame(data = matched_fluor_lines, columns = column_names)
    
    # making list in the same order as dataframe
    line_name_int = matched_df['Emission Line'].tolist()
    energy_match = matched_df['Energy (eV)'].tolist()
    
    # Removing repeats and averaging fluor line of elements with same emission lines (i.e. averaging Ce_Ka1, Ce_Ka2, etc. to make single peak representing Ce_Ka)
    unique_peak = matched_df['Peak #'].unique()
    matched_peaks = []
    matched_energy = []
    line_name = []
    matched_element = []
    for i in unique_peak:
        idx_peak = matched_df['Peak #'] == i
        peak_elements = matched_df['Element'][idx_peak]
        for j in set(peak_elements):
            idx_element = matched_df['Element'] == j
            idx_peak_element = idx_peak & idx_element
            
            if sum(idx_peak_element) > 1: 
                matched_peaks.append(i)
                matched_element.append(j)
                energy_int = list(compress(energy_match,idx_peak_element))
                matched_energy.append(np.mean(energy_int))
                line_names = list(compress(line_name_int,idx_peak_element))
                line_name.append(line_names[0][:-1])
            if sum(idx_peak_element) == 1:
                matched_peaks.append(i)
                matched_element.append(j)
                energy_int = list(compress(energy_match,idx_peak_element))
                matched_energy.append(np.mean(energy_int))
                line_name.extend(list(compress(line_name_int,idx_peak_element)))
    
    
    # Output list of matched elements, the fluorescence line name, and the energy (eV)
    matched_fluor_lines = sorted([list(a) for a in zip(matched_peaks, matched_element, line_name, matched_energy)], key=lambda l:l[3])
    
    column_names =  ["Peak #", "Element", "Emission Line", "Energy (eV)"]
    matched_df = pd.DataFrame(data = matched_fluor_lines, columns = column_names)

    
    return matched_fluor_lines, matched_df


########## Defining Gaussians ##########
# Define the Gaussian function
def gaussian(x, amplitude, mean, std_dev):
    return amplitude * np.exp(-(x - mean)**2 / (2 * std_dev**2))



########## define the sum of multiple gaussians ##########
def multi_gaussians(x, *params):
    num_gaussians = len(params) // 3
    result = np.zeros_like(x)
    
    for i in range(num_gaussians):
        amp, mean, stddev = params[i*3 : (i+1)*3]
        result += gaussian(x, amp, mean, stddev)
    
    return result

########## Fit background ##########
## Function to fit spectra background
# arpls approach to fit background
r"""
    Baseline correction using asymmetrically
    reweighted penalized least squares smoothing
    Sung-June Baek, Aaron Park, Young-Jin Ahna and Jaebum Choo,
    Analyst, 2015, 140, 250 (2015)

    Abstract

    Baseline correction methods based on penalized least squares are successfully
    applied to various spectral analyses. The methods change the weights iteratively
    by estimating a baseline. If a signal is below a previously fitted baseline,
    large weight is given. On the other hand, no weight or small weight is given
    when a signal is above a fitted baseline as it could be assumed to be a part
    of the peak. As noise is distributed above the baseline as well as below the
    baseline, however, it is desirable to give the same or similar weights in
    either case. For the purpose, we propose a new weighting scheme based on the
    generalized logistic function. The proposed method estimates the noise level
    iteratively and adjusts the weights correspondingly. According to the
    experimental results with simulated spectra and measured Raman spectra, the
    proposed method outperforms the existing methods for baseline correction and
    peak height estimation.

    Inputs:
        y:
            input data (i.e. chromatogram of spectrum)
        lam:
            parameter that can be adjusted by user. The larger lambda is,
            the smoother the resulting background, z
        ratio:
            wheighting deviations: 0 < ratio < 1, smaller values allow less negative values
        itermax:
            number of iterations to perform
    Output:
        the fitted background vector

    """

def arpls(y, lam=1e4, ratio=0.01, itermax=1000):
    
    N = len(y)
#  D = sparse.csc_matrix(np.diff(np.eye(N), 2))
    D = sparse.eye(N, format='csc')
    D = D[1:] - D[:-1]  # numpy.diff( ,2) does not work with sparse matrix. This is a workaround.
    D = D[1:] - D[:-1]

    H = lam * D.T * D
    w = np.ones(N)
    for i in range(itermax):
        W = sparse.diags(w, 0, shape=(N, N))
        WH = sparse.csc_matrix(W + H)
        C = sparse.csc_matrix(cholesky(WH.todense()))
        z = spsolve(C, spsolve(C.T, w * y))
        d = y - z
        dn = d[d < 0]
        m = np.mean(dn)
        s = np.std(dn)
        wt = 1. / (1 + np.exp(2 * (d - (2 * s - m)) / s))
        if np.linalg.norm(w - wt) / np.linalg.norm(w) < ratio:
            break
        w = wt
    return z


## Find mean square error
def mean_square_error(observed, predicted):
    observed = np.array(observed)
    predicted = np.array(predicted)

    return np.mean((observed - predicted)**2)


## Spectra Fitting Function
def peak_fitting(x, y, peaks, window):
    # Fit background data
    bkg_fit = arpls(y)

    # Fit Gaussian to each peak
    peak_tails = []
    combined_gaussian_fit = np.zeros_like(x)
    popt_all = []
    y_bkg_subtracted = y - bkg_fit

    for peak_index in peaks:
        x_peak = x[max(0, peak_index - window): min(len(x), peak_index + window)]
        y_peak = y_bkg_subtracted[max(0, peak_index - window): min(len(y), peak_index + window)]

        # Fit Gaussian
        popt, _ = curve_fit(gaussian, x_peak, y_peak, p0=[max(y_peak), x[peak_index], 1.0], maxfev = int(1e7))
        popt_all.append(popt)

        # Compile Gaussians of each peak 
        combined_gaussian_fit += gaussian(x, *popt)

    
    # Set up while loop to refine fit until convergence
    tolerance = 1e9
    error = tolerance + 1
    TIMEOUT = 60 # set maximum allowed time for while loop refinement to run in seconds
    start = perf_counter() # start timer

    while error > tolerance: 
        # Fit results to multi-gaussian function
        popt, _ = curve_fit(multi_gaussians, x, y_bkg_subtracted, p0 = [popt_all], maxfev = int(1e7))
        multi_gaussian_fit = multi_gaussians(x, *popt)
        
        error = mean_square_error(y, multi_gaussian_fit)

        popt_all = popt

        if perf_counter() - start > TIMEOUT:
            break

    peak_fit = multi_gaussian_fit + bkg_fit

    return peak_fit, bkg_fit, popt_all
    

########## AOI Analysis ##########
## Function to determine optimum paramters to extract Detector Area of Interest Spectrum
# * Inputs
#     1. filename: file path to hdf5 (.h5) file containing total scan data
#     2. min_energy: minimum of energy range of interest (keV)
#     3. elements: anticipated elements contained in the sample from background analysis using pyXRF
# * Outputs
#     1. detector_2D_map_fig: HTML figure containing 2D map of detector
#     2. fig1: HTML figure containing all relevant data/information processed that
#        can be used later to plot the results        
#     3. peak_fit_params: parameters used to define the gaussian fit of peaks in background subtracted partilce spectrum
def AOI_particle_analysis(filename, min_energy, elements):
    ########## Load data filenin variable ##########
    with h5py.File(filename, 'r') as file:
        data = file['xrfmap/detsum/counts'][:]
        pos_data = file['xrfmap/positions/pos'][:]
        group_name = 'xrfmap/scan_metadata'
        if group_name in file:
            group = file[group_name]
            attributes = dict(group.attrs)
            incident_energy = attributes['instrument_mono_incident_energy'] # keV
        else:
            print(f"Group '{group_name}' not found in the HDF5 file.")
    
    # Position axes
    x_pos = np.linspace(pos_data[0].min(),pos_data[0].max(),data.shape[1])
    y_pos = np.linspace(pos_data[1].min(),pos_data[1].max(),data.shape[1])

    # Use incident X-ray energy to define energy range of interest 
    max_energy = incident_energy + 0.8 # incident energy
    energy = 0.01*np.arange(data.shape[2])
    min_idx = max([i for i, v in enumerate(energy) if v <= min_energy])
    max_idx = min([i for i, v in enumerate(energy) if v >= max_energy])


    # Total summed spectrum
    sum_data = np.sum(data, axis = (0,1))
    sum_data = sum_data[min_idx:max_idx]
    
    ########## Plotting whole detector view to identify AOI ##########
    temp = np.sum(data,axis = (2))
    
    detector_2D_map_fig = go.Figure(data = go.Heatmap(z = temp, colorscale = 'Viridis', colorbar = {'exponentformat': 'e'}))
    detector_2D_map_fig.update_layout(title_text = 'Summed XRF Map for <br>' + filename[-26:-13]+' @ '+str(incident_energy)+' keV', 
                                      title_x = 0.5,
                                      width = 500,
                                      height = 500,
                                      font = dict(size = 20),
                                      xaxis = dict(title = 'X-axis'),
                                      yaxis = dict(title = 'Y-axis'))
    
    detector_2D_map_fig.show()

    ########## Handling bad pixels ##########
    user_input = input("Smooth over bad pixels? (Yes or No):")
    if user_input.lower() == "yes":
        # get number of values to extract
        user_input = input("Input integer value for number of bad pixels based on number unique xy coordinates showing distinctly lower intensity:")
        k = int(user_input) # number of values to be extracted 
        idx_flat = np.argpartition(temp.flatten(),k)[:k] # index of k lowest values 
        idx_2d = np.unravel_index(idx_flat,temp.shape)
        temp[idx_2d] = np.mean(temp) # new detecotr data without dead pixels 

        # plot new data
        detector_2D_map_fig = go.Figure(data = go.Heatmap(z = temp, colorscale = 'Viridis', colorbar = {'exponentformat': 'e'}))
        detector_2D_map_fig.update_layout(title_text = 'Summed XRF Map for <br>' + filename[-26:-13]+' @ '+str(incident_energy)+' keV', 
                                          title_x = 0.5,
                                          width = 500,
                                          height = 500,
                                          font = dict(size = 20),
                                          xaxis = dict(title = 'X-axis'),
                                          yaxis = dict(title = 'Y-axis'))
        
        detector_2D_map_fig.show()
        

    ######### Selecting area of interest based on PyXRF mappings ##########
    # # y-direction
    user_input = input("Utilizing the detector map outputted, enter x values for area of interest (AOI) in slice format (e.g., '1:5'):")
    detector_ROI_columns = input_to_slice(user_input)
    detector_ROI_columns = slice(detector_ROI_columns.start+1, detector_ROI_columns.stop+1)
    
    
    # # x-direction
    user_input = input("Utilizing the detector map outputted, enter y values for area of interest (AOI) in slice format (e.g., '1:5'):")
    detector_ROI_rows = input_to_slice(user_input)
    detector_ROI_rows = slice(detector_ROI_rows.start,detector_ROI_rows.stop)
    
    
    AOI_data = data[detector_ROI_rows, detector_ROI_columns, :]
    y_int = y_pos[detector_ROI_columns]
    x_int = x_pos[detector_ROI_rows]
 
   
    # Sum spectrum in selected area
    AOI = np.sum(AOI_data, axis=(0,1))
    AOI = AOI[min_idx:max_idx]
    energy_int = energy[min_idx:max_idx]
    
    

    ######### Selecting background area based on PyXRF mappings ##########

    # # y-direction
    user_input = input("Utilizing the detector map outputted, enter x values for area containing background spectra in slice format (e.g., '1:5'):")
    detector_ROI_columns = input_to_slice(user_input)
    detector_ROI_columns = slice(detector_ROI_columns.start+1, detector_ROI_columns.stop+1)
    
    # # x-direction
    user_input = input("Utilizing the detector map outputted, enter y values for area containing background spectra in slice format (e.g., '1:5'):")
    detector_ROI_rows = input_to_slice(user_input)
    

    # identify background spectrum
    bkg_data = data[detector_ROI_rows, detector_ROI_columns, :]
 
 
   
    # Sum background spectrum in selected area
    background = np.sum(bkg_data, axis=(0,1))
    background = background[min_idx:max_idx]


    # Extracting only the data up to the compton peak
    max_idx = np.argmax(AOI) # setting max_idx to be the max intensity in the spectrum which should be the compton peak
    AOI = AOI[:max_idx]
    background = background[:max_idx]
    max_energy = energy_int[max_idx]
    energy_int = energy_int[:max_idx]
    sum_data = sum_data[:max_idx]
    

    
    # Background subtracted AOI
    baseline = arpls(background) # Baseline of AOI spectrum
    AOI_bkg_sub = AOI - background
    AOI_bkg_sub[AOI_bkg_sub <= 0] = 0

    # add baseline to AOI spectrum
    AOI_bkg_sub = AOI_bkg_sub + baseline
    
    


    ########## Find peaks in data using parameter thresholds ##########
    prom = 70
    tall = 70
    dist = 10
    peaks, properties = find_peaks(AOI_bkg_sub, prominence = prom, height = tall, distance = dist)

   
    
    labels = []
    for i in range(len(peaks)):
        labels.extend(['Peak '+str(i+1)])
    
    ########## Spectra Plotting ##########
    # Plot raw AOI Spectrum
    fig1 = go.Figure(data = go.Scatter(x = energy_int, y = AOI, mode = 'lines', name = 'AOI Spectra'), layout_xaxis_range = [min_energy,max_energy])
    fig1.update_layout(title = 'raw AOI Spectrum for '+filename[-26:-13],
                       width = 1600,
                       height = 800,
                       font = dict(size = 20))
    
    # Plot Background subtracted AOI spectrum
    fig1.add_trace(go.Scatter(x = energy_int, y = AOI_bkg_sub, mode = 'lines', name = 'AOI bkg subtracted'))
    
    # Plot total summed spectrum 
    fig1.add_trace(go.Scatter(x = energy_int, y = sum_data, mode = 'lines', name = 'Summed Spectra'))

    # Plot background spectrum
    fig1.add_trace(go.Scatter(x = energy_int, y = background, mode = 'lines', name = 'Background Spectra'))

    # Plot baseline spectrum
    fig1.add_trace(go.Scatter(x = energy_int, y = baseline, mode = 'lines', name = 'Baseline Spectra'))
    
    # Plot points identified as peaks
    fig1.add_trace(go.Scatter(x = energy_int[peaks], y = AOI_bkg_sub[peaks], mode = 'markers+text', name = 'Peak fit', text = labels))
    
    # Plot formatting
    fig1.update_yaxes(title_text = 'Intensity (counts)', type = 'log', exponentformat = 'E')
    fig1.update_xaxes(title_text = 'Energy (keV)')
    fig1.update_traces(line={'width': 5})
    fig1.show()

    
    ########## Adjusting peak finding as needed ##########
    peak_props = input('Change peak thresholds for prominence, height, and/or distance (Yes or No)?')
    while True:
        if peak_props.lower() == 'no':
            break
        if peak_props.lower() == 'yes':
            user_input = input("Enter new values for prominence (" + str(prom) + "), height(" + str(tall) + "), and distance(" + str(dist) + ") (comma-separated), 'no' to exit: ")
            if user_input.lower() == 'no':
                break
                
            try: 
                prom, tall, dist = map(int, user_input.split(','))
            except ValueError:
                print("Invalid input. Please enter integers separated by a comma or 'no' to exit.")
                continue
            
            # Find peaks in data
            peaks, properties = find_peaks(AOI_bkg_sub, prominence = prom, height = tall, distance = dist)
            
            # Label peaks
            labels = []
            for i in range(len(peaks)): labels.extend(['Peak '+str(i+1)])
            
            # Creating new figure for AOI Spectrum
            fig1 = go.Figure(data = go.Scatter(x = energy_int, y = AOI, mode = 'lines', name = 'AOI Spectra'), layout_xaxis_range = [min_energy,max_energy])
            fig1.update_layout(title = 'AOI Spectrum for '+filename[-26:-13],
                               width = 1600,
                               height = 800,
                               font = dict(size = 20))
            
            # Plot Background subtracted AOI spectrum
            fig1.add_trace(go.Scatter(x = energy_int, y = AOI_bkg_sub, mode = 'lines', name = 'AOI bkg subtracted'))
                        
            # Plot total summed spectrum 
            fig1.add_trace(go.Scatter(x = energy_int, y = sum_data, mode = 'lines', name = 'Summed Spectra'))

            # Plot background spectrum
            fig1.add_trace(go.Scatter(x = energy_int, y = background, mode = 'lines', name = 'Background Spectra'))

            # Plot baseline spectrum
            fig1.add_trace(go.Scatter(x = energy_int, y = baseline, mode = 'lines', name = 'Baseline Spectra'))
    
            # Plot points identified as peaks
            fig1.add_trace(go.Scatter(x = energy_int[peaks], y = AOI_bkg_sub[peaks],mode = 'markers+text', name = 'Peak fit', text = labels))

            # Plot formatting
            fig1.update_yaxes(title_text = 'Intensity (counts)', type = 'log', exponentformat = 'E')
            fig1.update_xaxes(title_text = 'Energy (keV)')
            fig1.update_traces(line={'width': 5})

            ########## Identify elements ##########
            # identify fluorescent line energy that most closely matches the determined peaks
            tolerance = 1.5 # allowed difference in percent
            matched_peaks, _ = identify_element_match(elements, energy_int[peaks]*1000, tolerance)
            # Plotting vertical lines for matched peaks and labeled with element symbol
            for i in range(len(matched_peaks)):
                fig1.add_vline(x = matched_peaks[i][3]/1000, line_width = 1.5, line_dash = 'dash', annotation_text = matched_peaks[i][1]+'_'+matched_peaks[i][2])
            fig1.show()

    
    
    ########## Fit spectra and plot results ##########
    user_input = input('Fit peaks (Yes or No)?')
    if user_input.lower() == 'yes':
        print('Beginning peak fitting')
        peak_fit, bkg_fit, peak_fit_params = peak_fitting(energy_int, AOI_bkg_sub, peaks, dist)
        

        # Find peaks in fitted data
        peaks, _ = find_peaks(peak_fit-bkg_fit)
        # Label peaks
        labels = []
        for i in range(len(peaks)): labels.extend(['Peak '+str(i+1)])

        ########## Final Plot ##########
        fig1 = go.Figure(data = go.Scatter(x = energy_int, y = AOI, mode = 'lines', name = 'AOI Spectra'), layout_xaxis_range = [min_energy,max_energy])
        fig1.update_layout(title = 'AOI Spectrum for '+filename[-26:-13],
                           width = 1600,
                           height = 800,
                           font = dict(size = 20))
        
        # Plot Background subtracted AOI spectrum
        fig1.add_trace(go.Scatter(x = energy_int, y = AOI_bkg_sub, mode = 'lines', name = 'AOI bkg subtracted'))
                    
        # Plot total summed spectrum 
        fig1.add_trace(go.Scatter(x = energy_int, y = sum_data, mode = 'lines', name = 'Summed Spectra'))

        # Plot background spectrum
        fig1.add_trace(go.Scatter(x = energy_int, y = background, mode = 'lines', name = 'Background Spectra'))

        # Plot baseline spectrum
        fig1.add_trace(go.Scatter(x = energy_int, y = baseline, mode = 'lines', name = 'Baseline Spectra'))

        # Plot peak and background fits
        fig1.add_trace(go.Scatter(x = energy_int, y = peak_fit, mode = 'lines', name ='AOI Spectra Fit'))
        fig1.add_trace(go.Scatter(x = energy_int, y = bkg_fit, mode = 'lines', name = 'AOI Spectra Bkg Fit'))

        # Plot points identified as peaks
        fig1.add_trace(go.Scatter(x = energy_int[peaks], y = peak_fit[peaks],mode = 'markers+text', name = 'Peak fit', text = labels))


        # Plot formatting
        fig1.update_yaxes(title_text = 'Intensity (counts)', type = 'log', exponentformat = 'E')
        fig1.update_xaxes(title_text = 'Energy (keV)')
        fig1.update_traces(line={'width': 5})

        ########## Identify elements ##########
        # identify fluorescent line energy that most closely matches the determined peaks
        tolerance = 2 # allowed difference in percent
        matched_peaks, _ = identify_element_match(elements, energy_int[peaks]*1000, tolerance)
        # Plotting vertical lines for matched peaks and labeled with element symbol
        for i in range(len(matched_peaks)):
            fig1.add_vline(x = matched_peaks[i][3]/1000, line_width = 1.5, line_dash = 'dash', annotation_text = matched_peaks[i][1]+'_'+matched_peaks[i][2])

        # show figure
        fig1.show()
    

    
    
    ########## XRF Map Plotting ##########
    # Defing energy range of interest from user input
    while True:
        user_input = input(str(len(peaks))+' peaks found. ''How many energy range(s) should be plotted in 2D? Enter 0 to exit')
        try: 
            ranges = int(user_input)
            break
        except ValueError:
            print("Invalid input. Please enter a single integer greater than 0 or enter '0' to exit.")
            continue

        
    for i in range(ranges):
        energy_range_str = input('Energy (keV*100) range ' + str(i+1) + ' to be plotted in 2D? (min:max+1)')
        energy_range = slice( *map(int, energy_range_str.split(':')))

        # Plot 2D map of AOI 
        d_element = AOI_data[:, :, energy_range]
        element_data = np.sum(d_element, axis=(2))

        fig, ax = plt.subplots()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size = '5%', pad = 0.05)
        
        ax.set_title('Energies in range: ' + str(round(energy[energy_range].min(),3)) + '-' + str(round(energy[energy_range].max(),3)) + ' keV')
        im = ax.imshow(element_data, extent = [x_int.min(), x_int.max(), y_int.min(), y_int.max()], cmap='viridis')
        
        
        fig.colorbar(im, cax=cax, orientation = 'vertical')
        plt.show()
        
        

    return detector_2D_map_fig, fig1, peak_fit_params



########## AOI Extraction ##########
## Function to Extract Detector Area of Interest Spectrum using previously determined parameters
# * Inputs
#     1. filename: file path to hdf5 (.h5) file containing total scan data
#     2. min_energy: minimum of energy range of interest (keV)
#     3. elements: anticipated elements contained in the sample from background analysis using pyXRF
# * Outputs
#     1. detector_2D_map_fig: HTML figure containing 2D map of detector
#     2. fig1: HTML figure containing all relevant data/information processed that
#        can be used later to plot the results        
#     3. peak_fit_params: parameters used to define the gaussian fit of peaks in background subtracted partilce spectrum
def AOI_extractor(filename, min_energy, elements, AOI_x, AOI_y, BKG_x, BKG_y, prom, height, dist, energy_range, bad_pixels):
    ########## Load data file in variable ##########
    with h5py.File(filename, 'r') as file:
        data = file['xrfmap/detsum/counts'][:]
        pos_data = file['xrfmap/positions/pos'][:]
        group_name = 'xrfmap/scan_metadata'
        if group_name in file:
            group = file[group_name]
            attributes = dict(group.attrs)
            incident_energy = attributes['instrument_mono_incident_energy'] # keV
        else:
            print(f"Group '{group_name}' not found in the HDF5 file.")
    
    

    ########## Use incident X-ray energy to define energy range of interest ##########
    max_energy = incident_energy + 0.8 # incident energy
    energy = 0.01*np.arange(data.shape[2])
    min_idx = max([i for i, v in enumerate(energy) if v <= min_energy])
    max_idx = min([i for i, v in enumerate(energy) if v >= max_energy])
    


    ########## Detector data ##########
    detector_data = np.sum(data,axis = (2))
    detector_2D_map_fig = go.Figure(data = go.Heatmap(z = detector_data, colorscale = 'Viridis', colorbar = {'exponentformat': 'e'}))
    detector_2D_map_fig.update_layout(title_text = 'Summed XRF Map for <br>' + filename[-26:-13]+' @ '+str(incident_energy)+' keV', 
                                      title_x = 0.5,
                                      width = 500,
                                      height = 500,
                                      font = dict(size = 20),
                                      xaxis = dict(title = 'X-axis'),
                                      yaxis = dict(title = 'Y-axis'))
    
    detector_2D_map_fig.show()

    
    k = bad_pixels # number of values to be extracted 
    idx_flat = np.argpartition(detector_data.flatten(),k)[:k] # index of k lowest values 
    idx_2d = np.unravel_index(idx_flat,detector_data.shape)
    detector_data[idx_2d] = np.mean(detector_data) # new detecotr data without dead pixels 

    # plot new data
    detector_2D_map_fig = go.Figure(data = go.Heatmap(z = detector_data, colorscale = 'Viridis', colorbar = {'exponentformat': 'e'}))
    detector_2D_map_fig.update_layout(title_text = 'Summed XRF Map for <br>' + filename[-26:-13]+' @ '+str(incident_energy)+' keV', 
                                      title_x = 0.5,
                                      width = 500,
                                      height = 500,
                                      font = dict(size = 20),
                                      xaxis = dict(title = 'X-axis'),
                                      yaxis = dict(title = 'Y-axis'))
    
    detector_2D_map_fig.show()
    
    ########## Total summed spectrum ##########
    sum_data = np.sum(data, axis = (0,1))
    sum_data = sum_data[min_idx:max_idx]
    
    
    ######### Setting area of interest ##########
    AOI_x = slice(AOI_x.start+1,AOI_x.stop+1)
    AOI_data = data[AOI_y, AOI_x, :]
    
    
    # Sum spectrum in selected area
    AOI = np.sum(AOI_data, axis=(0,1))
    AOI = AOI[min_idx:max_idx]
    energy_int = energy[min_idx:max_idx]
    
    
    ########## Position axes ##########
    # whole positions
    x_pos = np.linspace(pos_data[0].min(),pos_data[0].max(),data.shape[1])
    y_pos = np.linspace(pos_data[1].min(),pos_data[1].max(),data.shape[1])

    # AOI positions
    y_int = y_pos[AOI_y]
    x_int = x_pos[AOI_x]
    

    ######### Setting background area ##########
    # identify background spectrum
    BKG_x = slice(BKG_x.start+1, BKG_x.stop+1)
    bkg_data = data[BKG_y, BKG_x, :]
    
    # Sum background spectrum in selected area
    background = np.sum(bkg_data, axis=(0,1))
    background = background[min_idx:max_idx]


    # Extracting only the data up to the compton peak
    max_idx = np.argmax(AOI) # setting max_idx to be the max intensity in the spectrum which should be the compton peak
    AOI = AOI[:max_idx]
    background = background[:max_idx]
    max_energy = energy_int[max_idx]
    energy_int = energy_int[:max_idx]
    sum_data = sum_data[:max_idx]
    

    # Background subtracted AOI
    baseline = arpls(background) # Baseline of AOI spectrum
    AOI_bkg_sub = AOI - background
    AOI_bkg_sub[AOI_bkg_sub <= 0] = 0

    

    # add baseline to AOI spectrum
    AOI_bkg_sub = AOI_bkg_sub + baseline
    

    ########## Find peaks in data using parameter thresholds ##########
    peaks, properties = find_peaks(AOI_bkg_sub, prominence = prom, height = height, distance = dist)
        

       
    
    ########## Fit spectra and plot results ##########
    print('Beginning peak fitting')
    peak_fit, bkg_fit, peak_fit_params = peak_fitting(energy_int, AOI_bkg_sub, peaks, dist)
        

    # Find peaks in fitted data
    peaks, _ = find_peaks(peak_fit-bkg_fit)
    # Label peaks
    labels = []
    for i in range(len(peaks)): labels.extend(['Peak '+str(i+1)])

    ########## Identify elements ##########
    # identify fluorescent line energy that most closely matches the determined peaks
    tolerance = 1.5 # allowed difference in percent
    matched_peaks = identify_element_match(elements, energy_int[peaks]*1000, tolerance)

    ########## Final Plot ##########
    fig1 = go.Figure(data = go.Scatter(x = energy_int, y = AOI, mode = 'lines', name = 'AOI Spectra'), layout_xaxis_range = [min_energy,max_energy])
    fig1.update_layout(title = 'AOI Spectrum for '+filename[-26:-13],
                       width = 1600,
                       height = 800,
                       font = dict(size = 20))
    
    # Plot Background subtracted AOI spectrum
    fig1.add_trace(go.Scatter(x = energy_int, y = AOI_bkg_sub, mode = 'lines', name = 'AOI bkg subtracted'))
                
    # Plot total summed spectrum 
    fig1.add_trace(go.Scatter(x = energy_int, y = sum_data, mode = 'lines', name = 'Summed Spectra'))

    # Plot background spectrum
    fig1.add_trace(go.Scatter(x = energy_int, y = background, mode = 'lines', name = 'Background Spectra'))

    # Plot baseline spectrum
    fig1.add_trace(go.Scatter(x = energy_int, y = baseline, mode = 'lines', name = 'Baseline Spectra'))

    # Plot peak and background fits
    fig1.add_trace(go.Scatter(x = energy_int, y = peak_fit, mode = 'lines', name ='AOI Spectra Fit'))
    fig1.add_trace(go.Scatter(x = energy_int, y = bkg_fit, mode = 'lines', name = 'AOI Spectra Bkg Fit'))

    # Plot points identified as peaks
    fig1.add_trace(go.Scatter(x = energy_int[peaks], y = peak_fit[peaks],mode = 'markers+text', name = 'Peak fit', text = labels))


    # Plot formatting
    fig1.update_yaxes(title_text = 'Intensity (counts)', type = 'log', exponentformat = 'E')
    fig1.update_xaxes(title_text = 'Energy (keV)')
    fig1.update_traces(line={'width': 5})

    ########## Identify elements ##########
    # identify fluorescent line energy that most closely matches the determined peaks
    tolerance = 1.5 # allowed difference in percent
    matched_peaks, _ = identify_element_match(elements, energy_int[peaks]*1000, tolerance)
    # Plotting vertical lines for matched peaks and labeled with element symbol
    for i in range(len(matched_peaks)):
        fig1.add_vline(x = matched_peaks[i][3]/1000, line_width = 1.5, line_dash = 'dash', annotation_text = matched_peaks[i][1]+'_'+matched_peaks[i][2])

    # show figure
    fig1.show()


    
    
    ########## XRF Map Plotting ##########
    # Defing energy range of interest from user input
    while True:
        user_input = input(str(len(peaks))+' peaks found. ''How many energy range(s) should be plotted in 2D? Enter 0 to exit')
        try: 
            ranges = int(user_input)
            break
        except ValueError:
            print("Invalid input. Please enter a single integer greater than 0 or enter '0' to exit.")
            continue

        
    for i in range(ranges):
        energy_range_str = input('Energy (keV*100) range ' + str(i+1) + ' to be plotted in 2D? (min:max+1)')
        energy_range = slice( *map(int, energy_range_str.split(':')))

        # Plot 2D map of AOI 
        d_element = AOI_data[:, :, energy_range]
        element_data = np.sum(d_element, axis=(2))

        fig, ax = plt.subplots()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size = '5%', pad = 0.05)
        
        ax.set_title('Energies in range: ' + str(round(energy[energy_range].min(),3)) + '-' + str(round(energy[energy_range].max(),3)) + ' keV')
        im = ax.imshow(element_data, extent = [x_int.min(), x_int.max(), y_int.min(), y_int.max()], cmap='viridis')
        
        
        fig.colorbar(im, cax=cax, orientation = 'vertical')
        plt.show()

   
   
    return detector_data, fig1, peak_fit_params, x_pos, y_pos, matched_peaks




########## Extract detector image data of selected file ##########
# * Inputs
#     1. filename: file path to hdf5 (.h5) file containing total scan data
# * Outputs
#     1. detector_data
#     2. x_pos, y_pos: x and y positions extracted from hdf5 file
def extract_detector_data(filename):
    ########## Load data file in variable ##########
    with h5py.File(filename, 'r') as file:
        data = file['xrfmap/detsum/counts'][:]
        pos_data = file['xrfmap/positions/pos'][:]
        group_name = 'xrfmap/scan_metadata'
        if group_name in file:
            group = file[group_name]
            attributes = dict(group.attrs)
            incident_energy = attributes['instrument_mono_incident_energy'] # keV
        else:
            print(f"Group '{group_name}' not found in the HDF5 file.")

    
    ########## Use incident X-ray energy to define energy range of interest ##########
    max_energy = incident_energy + 0.8 # incident energy
    energy = 0.01*np.arange(data.shape[2])
    min_idx = max([i for i, v in enumerate(energy) if v <= min_energy])
    max_idx = min([i for i, v in enumerate(energy) if v >= max_energy])

    ########## Position axes ##########
    # whole positions
    x_pos = np.linspace(pos_data[0].min(),pos_data[0].max(),data.shape[1])
    y_pos = np.linspace(pos_data[1].min(),pos_data[1].max(),data.shape[1])
    
    ########## Detector data ##########
    detector_data = np.sum(data,axis = (2))
    detector_2D_map_fig = go.Figure(data = go.Heatmap(z = detector_data, colorscale = 'Viridis', colorbar = {'exponentformat': 'e'}))
    detector_2D_map_fig.update_layout(title_text = 'Summed XRF Map for <br>' + filename[-26:-13]+' @ '+str(incident_energy)+' keV', 
                                      title_x = 0.5,
                                      width = 500,
                                      height = 500,
                                      font = dict(size = 20),
                                      xaxis = dict(title = 'X-axis'),
                                      yaxis = dict(title = 'Y-axis'))
    
    detector_2D_map_fig.show()
    
    ########## Handling bad pixels ##########
    user_input = input("Smooth over bad pixels? (Yes or No):")
    if user_input.lower() == "yes":
        # get number of values to extract
        user_input = input("Input integer value for number of bad pixels based on number unique xy coordinates showing distinctly lower intensity:")
        k = int(user_input) # number of values to be extracted 
        idx_flat = np.argpartition(detector_data.flatten(),k)[:k] # index of k lowest values 
        idx_2d = np.unravel_index(idx_flat,detector_data.shape)
        detector_data[idx_2d] = np.mean(detector_data) # new detecotr data without dead pixels 
    
        # plot new data
        detector_2D_map_fig = go.Figure(data = go.Heatmap(z = detector_data, colorscale = 'Viridis', colorbar = {'exponentformat': 'e'}))
        detector_2D_map_fig.update_layout(title_text = 'Summed XRF Map for <br>' + filename[-26:-13]+' @ '+str(incident_energy)+' keV', 
                                          title_x = 0.5,
                                          width = 500,
                                          height = 500,
                                          font = dict(size = 20),
                                          xaxis = dict(title = 'X-axis'),
                                          yaxis = dict(title = 'Y-axis'))
        
        detector_2D_map_fig.show()
    
    return detector_data, x_pos, y_pos

