import pandas as pd
import pyrfume
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

from final.globals import CID_INDEX_HEADER, ODOR_DILUTION

molecules = pyrfume.load_data("keller_2016/molecules.csv", remote=True)


## https://github.com/zinph/Cheminformatics/blob/master/compute_descriptors/RDKit_2D.py


class RDKit_2D:
    def __init__(self, smiles):
        self.mols = [Chem.MolFromSmiles(i) for i in smiles]
        self.smiles = smiles

    def compute_2Drdkit(self):
        rdkit_2d_desc = []
        calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
        header = calc.GetDescriptorNames()
        for i in range(len(self.mols)):
            ds = calc.CalcDescriptors(self.mols[i])
            rdkit_2d_desc.append(ds)
        df = pd.DataFrame(rdkit_2d_desc, columns=header)
        return df


def load_molecular_descriptors() -> (pd.DataFrame, []):
    """
    Breakdown of molecules to atoms and properties.
    Returns: df with the molecular description of the molecules, list of the features
    """
    print('Loading molecular descriptors data')
    molecular_descriptors = RDKit_2D(molecules.IsomericSMILES)
    df_molecular_description = molecular_descriptors.compute_2Drdkit()
    df_molecular_description[CID_INDEX_HEADER] = molecules.index.values

    # remove columns where all values are zero
    df_molecular_description = df_molecular_description.loc[:, (df_molecular_description != 0).any(axis=0)]

    feature_names = get_feature_names(df_molecular_description)

    return df_molecular_description, feature_names


def get_feature_names(molecular_descriptors):
    """
    Get list of the feature names.
    Args:
        molecular_descriptors: df with the features

    Returns: list of headers
    """
    feature_names = list(molecular_descriptors.columns)
    feature_names.remove(CID_INDEX_HEADER)
    feature_names.append(ODOR_DILUTION)
    return feature_names
