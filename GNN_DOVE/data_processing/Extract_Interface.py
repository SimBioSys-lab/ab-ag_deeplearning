
# Publication:  "Protein Docking Model Evaluation by Graph Neural Networks", Xiao Wang, Sean T Flannery and Daisuke Kihara,  (2020)


#GNN-Dove is a computational tool using graph neural network that can evaluate the quality of docking protein-complexes.


#Copyright (C) 2020 Xiao Wang, Sean T Flannery, Daisuke Kihara, and Purdue University.


#License: GPL v3 for academic use. (For commercial use, please contact us for different licensing.)


#Contact: Daisuke Kihara (dkihara@purdue.edu)


#


# This program is free software: you can redistribute it and/or modify


# it under the terms of the GNU General Public License as published by


# the Free Software Foundation, version 3.


#


# This program is distributed in the hope that it will be useful,


# but WITHOUT ANY WARRANTY; without even the implied warranty of


# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the


# GNU General Public License V3 for more details.


#


# You should have received a copy of the GNU v3.0 General Public License


# along with this program.  If not, see https://www.gnu.org/licenses/gpl-3.0.en.html.


import os
from ops.Timer_Control import set_timeout,after_timeout
RESIDUE_Forbidden_SET={"FAD"}


import numpy as np

def Extract_Interface(pdb_path):
    """
    Specially for 2 docking models
    :param pdb_path: Docking model path
    :return: Paths to receptor and ligand interface files
    """
    receptor_list = []
    ligand_list = []
    rlist = []
    llist = []
    count_r = 0
    count_l = 0

    with open(pdb_path, 'r') as file:
        line = file.readline()
        
        # Find the first 'ATOM' line
        while line[0:4] != 'ATOM':
            line = file.readline()
        
        b = 0
        tmp_list = []

        while line:
            dat_in = line[0:80].split()

            if len(dat_in) == 0:
                line = file.readline()
                continue

            if dat_in[0] == 'TER':
                b += 1
            elif dat_in[0] == 'ATOM':
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                atom_type = line[13:16].strip()

                if b == 0:
                    rlist.append([x, y, z, atom_type, count_r])
                    receptor_list.append(line)
                    count_r += 1
                else:
                    llist.append([x, y, z, atom_type, count_l])
                    ligand_list.append(line)
                    count_l += 1

            line = file.readline()

    print(f"Extracting {len(receptor_list)} atoms for receptor, {len(ligand_list)} atoms for ligand")
    
    final_receptor, final_ligand = Form_interface(np.array(rlist), np.array(llist), receptor_list, ligand_list)
    
    rpath = Write_Interface(final_receptor, pdb_path, ".rinterface")
    lpath = Write_Interface(final_ligand, pdb_path, ".linterface")
    
    print(rpath, lpath)
    return rpath, lpath

@set_timeout(100000, after_timeout)
def Form_interface(rlist, llist, receptor_list, ligand_list, cut_off=10):
    cut_off_sq = cut_off ** 2

    # Extract only the coordinates (x, y, z) for both receptor and ligand
    r_coords = rlist[:, :3].astype(float)
    l_coords = llist[:, :3].astype(float)

    # Compute pairwise squared Euclidean distance between all receptor and ligand atoms
    dist_matrix = np.sum((r_coords[:, np.newaxis, :] - l_coords[np.newaxis, :, :]) ** 2, axis=-1)

    # Get indices where distance is below the cutoff
    close_contact_indices = np.where(dist_matrix <= cut_off_sq)

    r_index = set(close_contact_indices[0])  # Receptor indices
    l_index = set(close_contact_indices[1])  # Ligand indices

    print(f"After filtering the interface region, {len(r_index)} residues in receptor, {len(l_index)} residues in ligand")
    
    # Extract the corresponding receptor and ligand lines based on indices
    final_receptor = [receptor_list[int(rlist[i, 4])] for i in r_index]
    final_ligand = [ligand_list[int(llist[i, 4])] for i in l_index]

    print(f"After filtering the interface region, {len(final_receptor)} receptor atoms, {len(final_ligand)} ligand atoms")
    
    return final_receptor, final_ligand




def Write_Interface(line_list,pdb_path,ext_file):
    new_path=pdb_path[:-4]+ext_file
    with open(new_path,'w') as file:
        for line in line_list:
            #check residue in the common residue or not. If not, no write for this residue
            residue_type = line[17:20]
            if residue_type in RESIDUE_Forbidden_SET:
                continue
            file.write(line)
    return new_path
