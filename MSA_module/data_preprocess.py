import numpy as np
import pandas as pd


class PreprocessMSAData:
    def __init__(
        self,
        a3m_file,
        train_set,
        test_set,
        combined_edge_lists="combined_edge_lists.npz",
        ss_file="secondary_structure_data.csv",
        sasa_file="sasa_data.csv",
        pt_file="cdrs_output.csv",
        max_len=1200,
        output_train_seq_file="preprocessed_train_sequences.npz",
        output_test_seq_file="preprocessed_test_sequences.npz",
        output_train_edge_file="train_edge_lists.npz",
        output_test_edge_file="test_edge_lists.npz",
    ):
        self.a3m_file = a3m_file
        self.train_set = set(train_set)
        self.test_set = set(test_set)
        self.combined_edge_lists = np.load(combined_edge_lists)
        self.ss_data = pd.read_csv(ss_file, index_col=0)
        self.sasa_data = pd.read_csv(sasa_file, index_col=0)
        self.pt_data = pd.read_csv(pt_file, index_col=0)
        self.max_len = max_len
        self.output_train_seq_file = output_train_seq_file
        self.output_test_seq_file = output_test_seq_file
        self.output_train_edge_file = output_train_edge_file
        self.output_test_edge_file = output_test_edge_file

        # Define the vocabulary (amino acids + padding and gap tokens)
        self.VOCAB = ["PAD", "A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V", "-"]
        self.token_to_idx = {token: idx for idx, token in enumerate(self.VOCAB)}

        # Initialize data holders
        self.train_sequences = []
        self.test_sequences = []
        self.train_identifiers = []
        self.test_identifiers = []
        self.train_edges = {}
        self.test_edges = {}

        # DataFrames for additional properties
        self.ss_train_df = pd.DataFrame()
        self.sasa_train_df = pd.DataFrame()
        self.pt_train_df = pd.DataFrame()
        self.ss_test_df = pd.DataFrame()
        self.sasa_test_df = pd.DataFrame()
        self.pt_test_df = pd.DataFrame()
        self.sapt_train_df = pd.DataFrame(columns=["sapt_value"])
        self.sapt_test_df = pd.DataFrame(columns=["sapt_value"])

        # Preprocess data
        self._preprocess_data()

    def _tokenize_and_pad(self, sequence):
        """
        Tokenizes and pads a sequence to the max length.
        """
        tokens = [self.token_to_idx.get(res, self.token_to_idx["PAD"]) for res in sequence]
        tokens = tokens[: self.max_len] + [self.token_to_idx["PAD"]] * (self.max_len - len(tokens))
        return tokens

    def _tokenize_secondary_structure(self, ss_sequence):
        token_map = {
            '-': 0,  # None
            'H': 1,  # Alpha helix
            'B': 2,  # Beta-bridge
            'E': 3,  # Strand
            'G': 4,  # 3-10 helix
            'I': 5,  # Pi helix
            'T': 6,  # Turn
            'S': 7   # Bend
        }
        tokenized = [token_map.get(char, 0) for char in ss_sequence]
        return tokenized[:self.max_len] + [-1] * (self.max_len - len(tokenized))

    def _pad_sasa(self, sasa_sequence):
        padded = sasa_sequence[:self.max_len] + [-1.0] * (self.max_len - len(sasa_sequence))
        return padded

    def _preprocess_data(self):
        """
        Main preprocessing function.
        """
        for chain_name in self.combined_edge_lists.files:
            pdbid = chain_name.split("_")[0]
            if pdbid in self.train_set:
                self._process_chain(chain_name, is_train=True)
            elif pdbid in self.test_set:
                self._process_chain(chain_name, is_train=False)

        # Save processed data
        self._save_sequences()
        self._save_edges()
        self._save_csv_files()

    def _process_chain(self, chain_name, is_train):
        """
        Process a single chain and split it into train or test data.
        """
        edge_list = self.combined_edge_lists[chain_name]
        tokenized_seq = self._load_sequence(chain_name)

        ss_sequence = self.ss_data.loc[chain_name].tolist()
        sasa_sequence = self.sasa_data.loc[chain_name].tolist()
        pt_sequence = self.pt_data.loc[chain_name].tolist()

        tokenized_ss = self._tokenize_secondary_structure(ss_sequence)
        padded_sasa = self._pad_sasa(sasa_sequence)
        sapt_value = sum(s for s, pt in zip(padded_sasa, pt_sequence) if pt == 1)

        ss_row = pd.DataFrame([tokenized_ss], index=[chain_name])
        sasa_row = pd.DataFrame([padded_sasa], index=[chain_name])
        pt_row = pd.DataFrame([pt_sequence], index=[chain_name])

        if is_train:
            self.train_sequences.append(tokenized_seq)
            self.train_identifiers.append(chain_name)
            self.train_edges[chain_name] = edge_list
            self.ss_train_df = pd.concat([self.ss_train_df, ss_row])
            self.sasa_train_df = pd.concat([self.sasa_train_df, sasa_row])
            self.pt_train_df = pd.concat([self.pt_train_df, pt_row])
            self.sapt_train_df.loc[chain_name] = [sapt_value]
        else:
            self.test_sequences.append(tokenized_seq)
            self.test_identifiers.append(chain_name)
            self.test_edges[chain_name] = edge_list
            self.ss_test_df = pd.concat([self.ss_test_df, ss_row])
            self.sasa_test_df = pd.concat([self.sasa_test_df, sasa_row])
            self.pt_test_df = pd.concat([self.pt_test_df, pt_row])
            self.sapt_test_df.loc[chain_name] = [sapt_value]

    def _load_sequence(self, chain_name):
        """
        Load sequence data from the A3M file for the given chain.
        """
        with open(self.a3m_file, "r") as f:
            for line in f:
                if line.startswith(f">{chain_name}"):
                    sequence = next(f).strip()
                    return self._tokenize_and_pad(sequence)
        raise ValueError(f"Chain {chain_name} not found in {self.a3m_file}")

    def _save_sequences(self):
        """
        Save sequences to separate files.
        """
        np.savez(
            self.output_train_seq_file,
            sequences=np.array(self.train_sequences),
            identifiers=self.train_identifiers,
        )
        np.savez(
            self.output_test_seq_file,
            sequences=np.array(self.test_sequences),
            identifiers=self.test_identifiers,
        )
        print("Sequences saved separately.")

    def _save_edges(self):
        """
        Save edges to separate files.
        """
        np.savez_compressed(self.output_train_edge_file, **self.train_edges)
        np.savez_compressed(self.output_test_edge_file, **self.test_edges)
        print("Edges saved separately.")

    def _save_csv_files(self):
        """
        Save CSV files for additional properties.
        """
        self.ss_train_df.to_csv("ss_train_data.csv", index=False)
        self.sasa_train_df.to_csv("sasa_train_data.csv", index=False)
        self.pt_train_df.to_csv("pt_train_data.csv", index=False)
        self.sapt_train_df.to_csv("sapt_train_data.csv", index=False)

        self.ss_test_df.to_csv("ss_test_data.csv", index=False)
        self.sasa_test_df.to_csv("sasa_test_data.csv", index=False)
        self.pt_test_df.to_csv("pt_test_data.csv", index=False)
        self.sapt_test_df.to_csv("sapt_test_data.csv", index=False)
        print("CSV files saved.")

# Read train and test sets
train_set = []
test_set = []

with open("train_set.txt", "r") as f:
    train_set = [line.strip() for line in f if line.strip()]  # Remove whitespace and empty lines

with open("test_set.txt", "r") as f:
    test_set = [line.strip() for line in f if line.strip()]  # Remove whitespace and empty lines

# Example usage with the PreprocessMSAData class
preprocessor = PreprocessMSAData(
    a3m_file="ab.ds0",
    train_set=train_set,
    test_set=test_set,
    combined_edge_lists="combined_edge_lists.npz",
    ss_file="secondary_structure_data.csv",
    sasa_file="sasa_data.csv",
    pt_file="cdrs_output.csv",
    max_len=1200,
    output_train_seq_file="preprocessed_seq_ab_train_1200.npz",
    output_test_seq_file="preprocessed_seq_ab_test_1200.npz",
    output_train_edge_file="train_edge_lists.npz",
    output_test_edge_file="test_edge_lists.npz",
)

