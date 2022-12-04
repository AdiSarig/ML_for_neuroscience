import numpy as np
import pandas as pd

from final.globals import ODOR_PERCEPT_COL, CID_INDEX_HEADER, ODOR_DILUTION, FREQUENCY_THRESHOLD, HIGH_DILUTION, \
    LOW_DILUTION, IGNOR_ODORS


def load_perception_data(file_url) -> pd.DataFrame:
    """
    Load the perception data
    Args:
        file_url: url for the data

    Returns: df with the perception data

    """
    print('Loading perception data')
    # To load the data openpyxl must be installed (pandas requirements)
    df = pd.read_excel(file_url, header=2)

    # replace odors with new col names (without space at the end)
    columns = df.columns
    columns = list(columns[:-20])
    columns.extend(ODOR_PERCEPT_COL)
    df.columns = columns

    return df


def explore_data(df: pd.DataFrame):
    """
    Explore data and get some statistics
    Args:
        df: df to explore
    """
    print('Explore data')
    df.info()
    pd.set_option('float_format', '{:g}'.format)
    df.describe()


def median_for_odors(perception_data: pd.DataFrame) -> pd.DataFrame:
    """
    For each molecule, in every dilution, look for the perception odors that were rated by number of subjects above
     the thresholds and calculate the median rating.
    Args:
        perception_data: df with perception data of all subjects

    Returns:
        medians_df: df with CID, dilution and median odors that pass the threshold
    """
    # create the results df
    headers = [CID_INDEX_HEADER, ODOR_DILUTION]
    headers.extend(ODOR_PERCEPT_COL)
    medians_df = pd.DataFrame(columns=headers)

    for cid in perception_data[CID_INDEX_HEADER].unique():
        cid_data = perception_data.loc[perception_data[CID_INDEX_HEADER] == cid]
        for dilution in cid_data[ODOR_DILUTION].unique():
            # get data for specific molecule in specific dilution
            dilut_data = cid_data.loc[cid_data[ODOR_DILUTION] == dilution]

            # get frequencies of odor percept ratings and choose the ones above the threshold
            frequencies = dilut_data[ODOR_PERCEPT_COL].count(axis=0)
            threshold_odors = frequencies[frequencies > FREQUENCY_THRESHOLD].index.tolist()

            # calculate medians
            medians = dilut_data[threshold_odors].median(axis=0, skipna=True)

            # dummy series and df to add new row into medians_df
            s = pd.Series([cid, dilution], index=[CID_INDEX_HEADER, ODOR_DILUTION])
            df2 = pd.DataFrame(s.append(medians)).transpose()
            medians_df = medians_df.append(df2)

    return medians_df


def convert_dilutions(df):
    """
    Convert the dilutions into a low/high dilution per molecule instead of a meaningless string.
    Args:
        df: df with CID and dilution information (such as perception df or medians df)

    Returns:
        the same df but with the dilution converted
    """
    for cid in df[CID_INDEX_HEADER].unique():
        cid_data = df.loc[df[CID_INDEX_HEADER] == cid]
        dilutions_num = []
        dilutions_str = []
        for dilution in cid_data[ODOR_DILUTION].unique():
            # parse dilution string
            split_dilution = dilution.split('/')
            split_dilution[1] = split_dilution[1].replace(',', '')
            # map dilution value and origin string
            dilutions_num.append(float(split_dilution[0]) / float(split_dilution[1]))
            dilutions_str.append(dilution)

        # assuming there are only 2 dilutions, this should find both
        min_dilution_ind = np.argmin(dilutions_num)
        max_dilution_ind = np.argmax(dilutions_num)

        # Convert dilution string with the appropriate value
        df.loc[(df[CID_INDEX_HEADER] == cid) & (
                df[ODOR_DILUTION] == dilutions_str[min_dilution_ind]), ODOR_DILUTION] = HIGH_DILUTION
        df.loc[(df[CID_INDEX_HEADER] == cid) & (
                df[ODOR_DILUTION] == dilutions_str[max_dilution_ind]), ODOR_DILUTION] = LOW_DILUTION

    return df


def get_max_observed_odor(df):
    """
    Find the odor with maximum observations.
    Args:
        df: df with perception ratings

    Returns: name of oder with maximum observations

    """
    odor_rating_count = 0
    max_rated_odor = None

    odors = [x for x in ODOR_PERCEPT_COL if x not in IGNOR_ODORS]
    for odor in odors:
        if df[odor].count() > odor_rating_count:
            max_rated_odor = odor
            odor_rating_count = df[odor].count()
    return max_rated_odor