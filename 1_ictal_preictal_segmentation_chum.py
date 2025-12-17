# Objective:
# take the preictal_ictal recordings from raw data folder and segment them into ictal and preictal periods.
# preictal period should have the same length as the ictal period.
from pathlib import Path
import mne
from os import chdir, listdir
import pandas as pd
from natsort import natsorted 
import os
from scipy.io import savemat, loadmat
import numpy as np
import matplotlib.pyplot as plt
import re



# SET THE RAW DATA DIRECTORY
DATASET_ID = "CHUM/"
data_dir = "/home/julieletallec/smb_share/Donnee/iEEG/RAW/" + DATASET_ID
save_dir = "/home/julieletallec/smb_share/Equip_Rech/LeTallec_Julie/processed_data3/" + DATASET_ID +"/sc_fc/"

Patients_list = ['Patient_01', 'Patient_02', 'Patient_04', 'Patient_07', 'Patient_08', 'Patient_09', 
                 'Patient_11', 'Patient_12', 'Patient_14', 'Patient_15', 'Patient_16', 'Patient_17', 
                 'Patient_21', 'Patient_22', 'Patient_23', 'Patient_25']

for PAT_ID in Patients_list:
    # raw recordings
    rawdata_dir = data_dir + PAT_ID + " 2" 
    chdir(rawdata_dir)
    
    # sort patient files.
    st_search = PAT_ID[0:6]
    ls_srt_found = [rawdata_dir + '/' + spt 
                for spt in os.listdir(rawdata_dir) 
                if st_search in spt]
    sorted_file_names = natsorted(ls_srt_found)

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

    # bad channels additional list based on the second inspection
    bad_ch_vis_ins = {}
    if PAT_ID == 'Patient_04':
        bad_ch_vis_ins[PAT_ID] = ['SEEG G240']
    elif PAT_ID == 'Patient_07':
        bad_ch_vis_ins[PAT_ID] = ['SEEG E110', 'SEEG E111', 'SEEG E112', 'SEEG E113', 'SEEG E114', 'SEEG E115', 'SEEG E116', 'SEEG E117', 'SEEG E118', 'SEEG E119', 'SEEG E120', 'SEEG E121', 'SEEG E122', 'SEEG E123', 'SEEG E124', 'SEEG T13']
    elif PAT_ID == 'Patient_08':
        bad_ch_vis_ins[PAT_ID] = ['SEEG F42']
    elif PAT_ID == 'Patient_09':
        bad_ch_vis_ins[PAT_ID] = ['SEEG F92']
    elif PAT_ID == 'Patient_14':
        bad_ch_vis_ins[PAT_ID] = ['SEEG E101', 'SEEG E102', 'SEEG E103', 'SEEG E104', 'SEEG E105', 'SEEG E106', 'SEEG E107', 'SEEG E108', 'SEEG E109', 'SEEG E110', 'SEEG E111', 'SEEG E112', 'SEEG E113', 'SEEG E114', 'SEEG E115', 'SEEG E116',
                                   'SEEG E117', 'SEEG E118', 'SEEG E119', 'SEEG E120', 'SEEG E121', 'SEEG E122', 'SEEG E123', 'SEEG E124', 'SEEG F18']
    elif PAT_ID == 'Patient_20':
        bad_ch_vis_ins[PAT_ID] = ['SEEG G148']
    elif PAT_ID == 'Patient_22':
        bad_ch_vis_ins[PAT_ID] = ['SEEG A23']
    elif PAT_ID == 'Patient_23':
        bad_ch_vis_ins[PAT_ID] = ['SEEG A12']
    elif PAT_ID == 'Patient_24':
        bad_ch_vis_ins[PAT_ID] = ['SEEG H21', 'SEEG H22']
    elif PAT_ID == 'Patient_25':
        bad_ch_vis_ins[PAT_ID] = ['SEEG H21', 'SEEG H22']
    

    # LOAD SEIZURE-BY-SEIZURE
    for seizure_num in seizure_lists[PAT_ID]:
        # load the seizure file
        seizure = mne.io.read_raw_edf(sorted_file_names[seizure_num], preload=True, encoding='latin1')
        Fs = seizure.info['sfreq']

        # print patient ID and seizure ID and the date and time of the recording
        print(f"Patient ID: {PAT_ID}")
        print(f"Seizure ID: {seizure_num}")
        print(f"Recording date: {seizure.info['meas_date']}")
        print(f"Recording time: {seizure.info['meas_date'].strftime('%H:%M:%S')}")

        # Prepare the event ids, corresponding to the seizure onset and offset for different datasets.
        if PAT_ID in ['Patient_01', 'Patient_08', 'Patient_09', 'Patient_11', 'Patient_12', 
                  'Patient_14', 'Patient_15', 'Patient_16', 'Patient_17', 'Patient_21', 
                  'Patient_22', 'Patient_23', 'Patient_25'] or \
           (PAT_ID == 'Patient_07' and seizure_num != 7) or \
           (PAT_ID == 'Patient_04' and (seizure_num == 0 or seizure_num == 3 or seizure_num == 5)):
            sz_onset_ann = 'Sz clin ON'
            sz_offset_ann = 'Sz clin OFF'
        elif PAT_ID == 'Patient_02' or \
            (PAT_ID == 'Patient_07' and seizure_num==7) or \
            (PAT_ID == 'Patient_04' and (seizure_num == 1 or seizure_num == 7)):
            sz_onset_ann = 'Seizure ON'
            sz_offset_ann = 'Seizure OFF'

        # load noisy channels' indeces
        bad_ch_dir = Path(os.path.join(data_dir, "bad_channels_mat"))
        # load the bad channels in .mat format
        patient_short = re.sub(r'Patient_0*(\d+)', r'P\1', PAT_ID)
        bad_ch_file = bad_ch_dir / f"bad_ch_{patient_short}.mat"
        if not bad_ch_file.exists():
            bad_ch = {'bad_ch': np.array([])}  # Set bad_ch to an empty structure
        else:
            bad_ch = loadmat(bad_ch_file)
        # get the bad channels. if the bad channel is empty, set it to an empty list.
        if bad_ch['bad_ch'].size == 0:
            bad_ch = []
        else:
            bad_ch = bad_ch['bad_ch'][0]

        # load channel names
        ch_names = seizure.ch_names
        # get the bad channels names
        noisy_ch_names = []
        for ch in bad_ch:
            # bad channel indices are 1-based vs 0-based in python.
            noisy_ch_names.append(ch_names[int(ch)-1])
        # select the SEEG channels only
        seeg_ch_names = []
        for ch in ch_names:
            if ch.startswith('SEEG'):
                seeg_ch_names.append(ch)
        # get the bad channels names that are SEEG channels
        bad_ch_names = []
        bad_ch_names = [ch for ch in noisy_ch_names if ch in seeg_ch_names]
        if PAT_ID in bad_ch_vis_ins and bad_ch_vis_ins[PAT_ID]:
            bad_ch_names.extend(bad_ch_vis_ins[PAT_ID])
        good_ch_names = list(set(seeg_ch_names) - set(bad_ch_names)) 
        # drop the bad channels
        seizure.pick_channels(good_ch_names)
        # get the events
        events = mne.events_from_annotations(seizure)[0]
        # get the event id
        event_id = mne.events_from_annotations(seizure)[1]
        # get the event id for seizure onset and offset
        sz_onset_id = event_id[sz_onset_ann]
        sz_offset_id = event_id[sz_offset_ann]

        # get the seizure epoch and the same lenght preictal epoch
        # get the seizure onset and offset in seconds
        seizure_onset = events[events[:, 2] == sz_onset_id, 0][0] / Fs
        seizure_offset = events[events[:, 2] == sz_offset_id, 0][0] / Fs
        # get the seizure duration
        seizure_duration = seizure_offset - seizure_onset


        # get the seizure epoch
        ictal_epoch = seizure.copy().crop(seizure_onset, seizure_offset)
        # get the preictal epoch
        preictal_epoch = seizure.copy().crop(0, seizure_onset)

        # ictal period
        ictal_save_dir = os.path.join(save_dir, 'ictal', PAT_ID)
        os.makedirs(ictal_save_dir, exist_ok=True)  # crée le dossier si besoin
        ictal_save_file = os.path.join(ictal_save_dir, f"{PAT_ID}_ictal_{seizure_num}_raw.fif")
        ictal_epoch.save(ictal_save_file, overwrite=True)

        # preictal period
        preictal_save_dir = os.path.join(save_dir, 'preictal', PAT_ID)
        os.makedirs(preictal_save_dir, exist_ok=True)  # crée le dossier si besoin
        preictal_save_file = os.path.join(preictal_save_dir, f"{PAT_ID}_preictal_{seizure_num}_raw.fif")
        preictal_epoch.save(preictal_save_file, overwrite=True)

