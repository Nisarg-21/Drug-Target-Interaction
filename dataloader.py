import pandas as pd
import torch.utils.data as data
import torch
import numpy as np
from functools import partial
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from utils import integer_label_protein  # iter1 - DEAD CODE (CO-06): unused, kept intentionally


class DTIDataset(data.Dataset):         #sets up fuunction for calucalting features and edges
    def __init__(self, list_IDs, df, max_drug_nodes=290):
        self.list_IDs = list_IDs
        self.df = df
        self.max_drug_nodes = max_drug_nodes

        self.atom_featurizer = CanonicalAtomFeaturizer()
        self.bond_featurizer = CanonicalBondFeaturizer(self_loop=True)
        self.fc = partial(smiles_to_bigraph, add_self_loop=True)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        index = self.list_IDs[index]
        smiles = self.df.iloc[index]['SMILES']

        v_d = self.fc(smiles=smiles, node_featurizer=self.atom_featurizer, edge_featurizer=self.bond_featurizer)
        actual_node_feats = v_d.ndata.pop('h')
        num_actual_nodes = actual_node_feats.shape[0]
        num_virtual_nodes = self.max_drug_nodes - num_actual_nodes
        virtual_node_bit = torch.zeros([num_actual_nodes, 1])
        actual_node_feats = torch.cat((actual_node_feats, virtual_node_bit), 1)
        v_d.ndata['h'] = actual_node_feats
        virtual_node_feat = torch.cat((torch.zeros(num_virtual_nodes, 74), torch.ones(num_virtual_nodes, 1)), 1)
        v_d.add_nodes(num_virtual_nodes, {"h": virtual_node_feat})
        # iter1 - FIXED (SW-07): was v_d.add_self_loop(), which adds a self-loop to *every* node.
        # Real atoms already got one from smiles_to_bigraph(add_self_loop=True) above, so they
        # ended up carrying two, skewing GCN degree normalisation. Only the padding nodes added
        # by add_nodes() lack one, so give a self-loop to exactly those and leave the atoms alone.
        # iter2 - FIXED: DGL 2.5 requires int32 node IDs. torch.arange() defaults to int64,
        # which add_edges() now rejects outright ("Expect argument u to have data type
        # torch.int32") instead of silently casting as older DGL did.
        virtual_node_ids = torch.arange(num_actual_nodes, self.max_drug_nodes, dtype=torch.int32)
        v_d.add_edges(virtual_node_ids, virtual_node_ids)

        v_p = self.df.iloc[index]['Protein']
        y = self.df.iloc[index]["Y"]

        return v_d, smiles, v_p, y


# iter1 - FIXED (C-01, C-02): this class was truncated mid-definition. The file ended on
# "return bat" (an undefined name) with no trailing newline, _get_nexts had no return
# statement, and there was no __iter__, __next__ or __len__ at all - so len() in
# Trainer.__init__ raised TypeError before a single batch was ever drawn. Rewritten in full.
class MultiDataLoader(object):  #pairs source and target batches together for CDAN training.
    def __init__(self, dataloaders, n_batches):               #restart the shorter loader
        if n_batches <= 0:
            raise ValueError("n_batches should be > 0")
        self._dataloaders = dataloaders
        self._n_batches = int(np.maximum(1, n_batches))
        self._init_iterators()

    def _init_iterators(self):
        self._iterators = [iter(dl) for dl in self._dataloaders]

    def _get_nexts(self):
        def _get_next_dl_batch(di, dl):
            try:
                batch = next(dl)
            except StopIteration:
                new_dl = iter(self._dataloaders[di])
                self._iterators[di] = new_dl
                batch = next(new_dl)
            return batch          # iter1 - FIXED (C-01): was "return bat"

        return [_get_next_dl_batch(di, dl) for di, dl in enumerate(self._iterators)]

    def __iter__(self):
        # Yields one collated batch per wrapped loader, for self._n_batches steps. The
        # shorter loader is restarted transparently by _get_next_dl_batch above.
        self._init_iterators()
        for _ in range(self._n_batches):
            yield self._get_nexts()

    def __next__(self):
        return self._get_nexts()

    # iter1 - FIXED (C-02): trainer.py calls len(self.train_dataloader) in __init__ and again
    # in train_da_epoch; without this the DA path died at Trainer construction.
    def __len__(self):
        return self._n_batches
