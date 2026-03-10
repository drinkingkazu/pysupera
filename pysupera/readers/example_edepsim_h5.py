from pysupera.utils import InteractionType
from enum import Enum, auto

class G4ProcessType(Enum):
    kProcessNotDefined = 0
    kProcessTransportation = 1
    kProcessElectromagnetic = 2 
    kProcessOptical = 3
    kProcessHadronic = 4
    kProcessPhotoLeptonHadron = 5
    kProcessDecay = 6
    kProcessGeneral = 7
    kProcessParameterization = 8
    kProcessUserDefined = 9

class G4ProcessSubtype(Enum):
    kSubtypeEMCoulombScattering = 1
    kSubtypeEMIonization = 2
    kSubtypeEMBremsstrahlung = 3
    kSubtypeEMPairProdByCharged = 4
    kSubtypeEMNuclearStopping = 8
    # EM subtypes for photons
    kSubtypeEMMultipleScattering = 10
    kSubtypeEMPhotoelectric = 12
    kSubtypeEMComptonScattering = 13
    kSubtypeEMGammaConversion = 14
    # Hadronic subtypes
    kSubtypeHadronElastic = 111
    kSubtypeHadronInelastic = 121
    kSubtypeHadronCapture = 131
    kSubtypeHadronChargeExchange = 161
    
    # General subtypes
    kSubtypeGeneralStepLimit = 401


def search_parents(A, B):

    sorter = np.argsort(A)
    idx = np.searchsorted(A, B, sorter=sorter)
    idx_clipped = np.clip(idx, 0, len(A) - 1)
    valid = A[sorter[idx_clipped]] == B
    return np.where(valid, sorter[idx_clipped], -1)


def get_parent_pdg(parts):

    parent_locs  = search_parents(parts['track_id'],parts['parent_track_id'])
    parent_valid = parent_locs != -1
    parent_zfill = np.where(parent_valid, parent_locs, 0)
    return np.where(parent_valid, parts['pdg'][parent_zfill], 0)
    
def get_interaction_type(parts,electron_energy_threshold=0.05):
    track_id    = parts['track_id']
    parent_track_id = parts['parent_track_id']
    pdg         = parts['pdg']
    parent_pdg  = get_parent_pdg(parts)
    proc_start  = parts['proc_start']
    subproc_start = parts['subproc_start']
    ke          = parts['ke']

    parent_locs  = search_parents(parts['track_id'],parts['parent_track_id'])
    parent_valid = parent_locs != -1
    parent_zfill = np.where(parent_valid, parent_locs, 0)
    #parent_track_id = np.where(parent_valid, parts['track_id'][parent_zfill], -1)

    xs, ys, zs = parts['x'], parts['y'], parts['z']
    dx = xs[parent_zfill] - xs
    dy = ys[parent_zfill] - ys
    dz = zs[parent_zfill] - zs
    dr = np.where(parent_valid, np.sqrt(dx**2 + dy**2 + dz**2), -1.0)

    itype = [InteractionType.kInvalidProcess]*len(parts)

    conditions = [(pdg == 2112),
                  (pdg > 1000000000),
                  (track_id == parent_track_id),
                  (pdg == 22),
                  (abs(pdg) == 11) & (proc_start == G4ProcessType.kProcessElectromagnetic) & (subproc_start == G4ProcessSubtype.kSubtypeEMPhotoelectric),
                  (abs(pdg) == 11) & (proc_start == G4ProcessType.kProcessElectromagnetic) & (subproc_start == G4ProcessSubtype.kSubtypeEMComptonScattering),
                  (abs(pdg) == 11) & (proc_start == G4ProcessType.kProcessElectromagnetic) & ((subproc_start == G4ProcessSubtype.kSubtypeEMGammaConversion) |
                                                                                              (subproc_start == G4ProcessSubtype.kSubtypeEMPairProdByCharged)),
                  (abs(pdg) == 11) & (proc_start == G4ProcessType.kProcessElectromagnetic) & (subproc_start == G4ProcessSubtype.kSubtypeEMIonization) & (abs(parent_pdg) == 22),
                  (abs(pdg) == 11) & (proc_start == G4ProcessType.kProcessElectromagnetic) & (subproc_start == G4ProcessSubtype.kSubtypeEMIonization) & (np.isin(abs(parent_pdg),[211,13,2212,321])),
                  (abs(pdg) == 11) & (proc_start == G4ProcessType.kProcessElectromagnetic) & (subproc_start == G4ProcessSubtype.kSubtypeEMIonization) & (abs(parent_pdg) == 22),
                  (abs(pdg) == 11) & (proc_start == G4ProcessType.kProcessElectromagnetic) & (subproc_start == G4ProcessSubtype.kSubtypeEMIonization),
                  (abs(pdg) == 11) & (proc_start == G4ProcessType.kProcessElectromagnetic),
                  (abs(pdg) == 11) & (proc_start == G4ProcessType.kProcessDecay),
                  (abs(pdg) == 11) & (proc_start == G4ProcessType.kProcessHadronic) & (subproc_start == 151) & (dr<0.0001) & (ke < electron_energy_threshold),
                  (abs(pdg) == 11) & (proc_start == G4ProcessType.kProcessHadronic) & (subproc_start == 151) & (dr<0.0001),
                  (abs(pdg) == 11) & (ke < electron_energy_threshold),
                  (abs(pdg) == 11),
                  ~(abs(pdg) == 11)]

    choices=[InteractionType.kNeutron,
             InteractionType.kNucleus,
             InteractionType.kPrimary,
             InteractionType.kPhoton,
             InteractionType.kPhotoElectron,
             InteractionType.kCompton, 
             InteractionType.kConversion,
             InteractionType.kIonization,
             InteractionType.kDelta,
             InteractionType.kCompton,
             InteractionType.kIonization,
             InteractionType.kInvalidProcess,
             InteractionType.kDecay,
             InteractionType.kIonization,
             InteractionType.kDecay,
             InteractionType.kCompton,
             InteractionType.kOtherShower,
             InteractionType.kTrack]
    
    return np.select(conditions, choices, default=InteractionType.kInvalidProcess)

def get_interaction_type_slow(parts, electron_energy_threshold=0.05):

    track_id    = parts['track_id']
    parent_id   = parts['parent_track_id']
    pdg         = parts['pdg']
    proc_start  = parts['proc_start']
    sproc_start = parts['subproc_start']
    ke          = parts['ke']

    parent_locs  = search_parents(parts['track_id'],parts['parent_track_id'])
    parent_valid = parent_locs != -1
    parent_zfill = np.where(parent_valid, parent_locs, 0)
    parent_pdg   = np.where(parent_valid, parts['pdg'][parent_zfill], 0)
    parent_track_id = np.where(parent_valid, parts['track_id'][parent_zfill], -1)
    
    xs, ys, zs = parts['x'], parts['y'], parts['z']
    dx = xs[parent_zfill] - xs
    dy = ys[parent_zfill] - ys
    dz = zs[parent_zfill] - zs
    dr = np.where(parent_valid, np.sqrt(dx**2 + dy**2 + dz**2), -1.0)

    itype = [InteractionType.kInvalidProcess]*len(parts)

    for idx in range(len(parts)):

        if pdg[idx] == 2112:
            itype[idx] = InteractionType.kNeutron

        elif pdg[idx] > 1000000000:
            itype[idx] = InteractionType.kNucleus

        elif track_id[idx] == parent_track_id[idx]:
            itype[idx] = InteractionType.kPrimary

        elif pdg[idx] == 22:
            itype[idx] = InteractionType.kPhoton

        elif abs(pdg[idx]) == 11:

            if proc_start[idx] == G4ProcessType.kProcessElectromagnetic:

                if subproc_start[idx] == G4ProcessSubtype.kSubtypeEMPhotoelectric:
                    itype[idx] = InteractionType.kPhotoElectron

                elif subproc_start[idx] == G4ProcessSubtype.kSubtypeEMComptonScattering:
                    itype[idx] = InteractionType.kCompton

                elif subproc_start[idx] == G4ProcessSubtype.kSubtypeEMComptonScattering:
                    itype[idx] = InteractionType.kConversion

                elif subproc_start[idx] == G4ProcessSubtype.kSubtypeEMPairProdByCharged:
                    itype[idx] = InteractionType.kConversion

                elif subproc_start[idx] == G4ProcessSubtype.kSubtypeEMPairProdByCharged:

                    if abs(parent_pdg[idx]) == 11:
                        itype[idx] = InteractionType.kIonization

                    elif abs(parent_pdg[idx]) in [211,13,2212,321]:
                        itype[idx] = InteractionType.kDelta

                    elif parent_pdg[idx] == 22:
                        itype[idx] = InteractionType.kCompton
                        
                    else:
                        itype[idx] = InteractionType.kIonization

                else:
                    raise ValueError

            elif proc_start[idx] == G4ProcessType.kProcessDecay:
                itype[idx] = InteractionType.kDecay

            elif proc_start[idx] == G4ProcessType.kProcessHadronic and subproc_start[idx] == 151 and dr<0.0001:
                if ke[idx] < electron_energy_threshold:
                    itype[idx] = InteractionType.kIonization
                else:
                    itype[idx] = InteractionType.kDecay

            else:
                if ke[idx] < electron_energy_threshold:
                    itype[idx] = InteractionType.kCompton
                else:
                    itype[idx] = InteractionType.kOtherShower
        else:
            itype[idx] = InteractionType.kTrack
                    
    return itype


from pysupera.data import Particle
import numpy as np
import time

PARTICLE_KEY='particle/geant4'
STEP_KEY='pstep/lar_vol'
ASS_KEY='ass/particle_pstep_lar_vol'
ELECTRON_ENERGY_THRESHOLD=0.05
Particle.time=0.
def read(fname='out_0100.h5',entries=None):
    local_time = 0.    
    with h5.File(fname,'r') as f:

        print(f.keys())

        if entries is None:
            entries = np.arange(len(f[PARTICLE_KEY]))

        parts_list=[]
        
        for entry in entries:
            t0=time.time()
            parts = f[PARTICLE_KEY][entry]
            steps = f[STEP_KEY][entry]
            ass   = f[ASS_KEY][entry]
            itype = np.array([t.value for t in get_interaction_type(parts,ELECTRON_ENERGY_THRESHOLD)],dtype=np.int32)
            num_parts = len(parts)
            local_time += time.time()-t0
            pysupera_parts = Particle.from_flat_arrays(parts['track_id'],
                                                       parts['parent_track_id'],
                                                       parts['root_track_id'],
                                                       parts['pdg'],
                                                       get_parent_pdg(parts),
                                                       itype,
                                                       steps,
                                                       np.column_stack([ass['start'][:num_parts],
                                                                        ass['end'  ][:num_parts]]),
                                                      )
            parts_list.append(pysupera_parts)
    print(local_time)
    return parts_list

import time
t0=time.time()
parts = read()
print(time.time()-t0,Particle.time)