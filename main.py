from final.build_models import build_models, explainability
from final.globals import ODOR_PERCEPT_COL, CID_INDEX_HEADER, PERCEPTION_URL
from final.molecular_descriptors import load_molecular_descriptors
from final.perception_data import load_perception_data, explore_data, median_for_odors, convert_dilutions, \
    get_max_observed_odor

if __name__ == '__main__':
    perception_data = load_perception_data(PERCEPTION_URL)
    molecular_descriptors, feature_names = load_molecular_descriptors()

    # convert perception data into medians df
    medians_df = median_for_odors(perception_data)
    medians_df = convert_dilutions(medians_df)

    # combine medians for odors with the molecular description (percept+molecular)
    df = medians_df.set_index(CID_INDEX_HEADER).join(molecular_descriptors.set_index(CID_INDEX_HEADER))
    explore_data(df[ODOR_PERCEPT_COL])

    # choose odor as the model target
    odor = get_max_observed_odor(df)
    print(f'Max observed odor is: {odor}')

    # build model
    model, mse, x_train = build_models(df, odor, feature_names)

    # try to explain the chosen model
    explainability(model, x_train)
