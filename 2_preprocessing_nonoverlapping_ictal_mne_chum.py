from normalize import normalize_windows
import mne
from pathlib import Path
from os import chdir, listdir
import pickle
import numpy as np


# SET THE DATA DIRECTORY
DATASET_ID = "CHUM"
data_dir = "//home/julieletallec/smb_share/Equip_Rech/LeTallec_Julie/processed_data3/" + DATASET_ID +"/sc_fc/ictal/"


# take the list of the patients.
Patients_list = listdir(data_dir)

#for PAT_ID in Patients_list[1:]:
for PAT_ID in Patients_list:
    
    # load ictal segments
    segdata_ictal_dir = data_dir + PAT_ID
    chdir(segdata_ictal_dir)

    # list of available seizure files per patient
    seizure_lists = {}
    if PAT_ID == 'Patient_01':
        seizure_lists[PAT_ID] = [0, 1, 2, 3, 4, 6, 7, 8, 10, 11] # 9 is exlucded, electrical seizure was 10 sec before clinical seizure. 5 is exluded since the annotations were not available
    elif PAT_ID == 'Patient_02':
        seizure_lists[PAT_ID] = [0, 1, 2, 3, 4]
    elif PAT_ID == 'Patient_04':
        seizure_lists[PAT_ID] = [0, 1, 3, 5, 7]
    elif PAT_ID == 'Patient_07':
        seizure_lists[PAT_ID] = [0, 1, 2, 3, 4, 5, 6, 7, 8] 
    elif PAT_ID == 'Patient_08':
        seizure_lists[PAT_ID] = [0] # 1,2,3,4 are excluded, pure electrical seizure
    elif PAT_ID == 'Patient_09':
        seizure_lists[PAT_ID] = [0, 1, 2, 3] 
    elif PAT_ID == 'Patient_11':
        seizure_lists[PAT_ID] = [0, 1, 2, 3, 4, 5, 6] 
    elif PAT_ID == 'Patient_12':
        seizure_lists[PAT_ID] = [0, 1, 2] 
    elif PAT_ID == 'Patient_14':
        seizure_lists[PAT_ID] = [0, 1, 2, 3] 
    elif PAT_ID == 'Patient_15':
        seizure_lists[PAT_ID] = [0] # 1,2 are excluded, pure electrical seizures
    elif PAT_ID == 'Patient_16':
        seizure_lists[PAT_ID] = [0, 1, 2, 3, 4]
    elif PAT_ID == 'Patient_17':
        seizure_lists[PAT_ID] = [0, 1, 2, 3]
    elif PAT_ID == 'Patient_21':
        seizure_lists[PAT_ID] = [0, 1, 2, 3, 4]
    elif PAT_ID == 'Patient_22':
        seizure_lists[PAT_ID] = [0, 1, 2, 3, 4]
    elif PAT_ID == 'Patient_23':
        seizure_lists[PAT_ID] = [0, 1]
    elif PAT_ID == 'Patient_25':
        seizure_lists[PAT_ID] = [0, 1, 2]
    
    # LOAD SEIZURE-BY-SEIZURE
    for seizure_num in seizure_lists[PAT_ID]:

        ictal_fname = PAT_ID + "_ictal_" + str(seizure_num) + "_raw.fif"
        ictal_data = mne.io.read_raw_fif(ictal_fname, preload=True)
        
        # PREPROCESSING TO NON-OVERLAPING WINDOWS
        fs = ictal_data.info['sfreq']
        ch_names = ictal_data.ch_names
        n_ch = len(ch_names)
        h_cutoff = 125
        l_cutoff = 0.5
        # 1. Apply low-pass filter
        filtered_data_ictal = ictal_data.copy().filter(l_freq=l_cutoff, h_freq=h_cutoff)
        # 2. Apply notch filters at 60Hz and 120Hz
        freqs = (60, 120)
        filtered_data_ictal = filtered_data_ictal.copy().notch_filter(freqs=freqs)
        # 3. Apply average reference
        referenced_data_ictal = filtered_data_ictal.copy().set_eeg_reference(ref_channels='average')
        # 4. Divide into windows
        window_size_sec = 1
        windows_ictal = mne.make_fixed_length_epochs(referenced_data_ictal, window_size_sec)
        # 5. Normalize windows
        windows_ictal_np = windows_ictal.get_data()
        normalize_windows = np.zeros_like(windows_ictal_np)
        for i, epoch in enumerate(windows_ictal_np):
            for ch in range(n_ch):
                normalize_windows[i, ch,:] = (epoch[ch,:] - np.mean(epoch[ch,:])) / np.std(epoch[ch,:])
        # create a new EpochsArray with the normalized windows
        normalized_windows_ictal = mne.EpochsArray(normalize_windows, windows_ictal.info)

        # 6. Save the normalized windows
        normalized_fname = PAT_ID + "_ictal_" + str(seizure_num) + "_processed.fif"
        normalized_windows_ictal.save(normalized_fname, overwrite=True)


