with open('interface_files', 'r') as itf_f:
    line = itf_f.readline()
    while line:
        interface = line.strip()
        pdb_name = interface.split('_')[0]
        chain_1 = interface.split('_')[2]
        chain_2 = interface.split('_')[3]
        
        # Process chain_1
        with open(f'/work/SimBioSys/Xing/data_collection/pdbs/test/a3m_DS/{pdb_name}_chain_{chain_1}.a3m.DS', 'r') as src, open('chain_1.ds', 'a') as dest:
            dest.write(f'< {pdb_name}_chain_{chain_1}\n')
            content = src.read()
            dest.write(content)
        
        # Process chain_2
        with open(f'/work/SimBioSys/Xing/data_collection/pdbs/test/a3m_DS/{pdb_name}_chain_{chain_2}.a3m.DS', 'r') as src, open('chain_2.ds', 'a') as dest:
            dest.write(f'< {pdb_name}_chain_{chain_2}\n')
            content = src.read()
            dest.write(content)
        
        # Process the interface file
        with open('/work/SimBioSys/Xing/data_collection/pdbs/test/interfaces/' + interface, 'r') as itf, open('tgt1.txt', 'a') as tgt1, open('tgt2.txt', 'a') as tgt2:
            interface_line = itf.readline()  # Use a different variable name
            while interface_line:
                if interface_line.strip() == f'chain_{chain_1}_residue_one_hot':
                    next_line = itf.readline()  # Use a different variable name
                    tgt1.write(next_line)
                if interface_line.strip() == f'chain_{chain_2}_residue_one_hot':
                    next_line = itf.readline()
                    tgt2.write(next_line)
                interface_line = itf.readline()  # Continue using the new variable
                
        line = itf_f.readline()  # Continue using the outer loop's line variable

