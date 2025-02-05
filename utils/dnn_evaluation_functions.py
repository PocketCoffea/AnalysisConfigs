import numpy as np
import awkward as ak


def get_dnn_prediction(session, input_name, output_name, events, variables):

    variables_array = []
    for var_name, attributes in variables.items():
        collection, feature = attributes

        if collection == "events":
            ak_array = getattr(events, feature)
        elif ":" in collection:
            ak_array = getattr(getattr(events, collection.split(":")[0]), feature)
            pos = int(collection.split(":")[1])
            ak_array = ak.fill_none(ak.pad_none(ak_array, pos + 1, clip=True), 0)[
                :, pos
            ]
        else:
            ak_array = getattr(events, collection)[feature]
        variables_array.append(
            np.array(
                ak.to_numpy(
                    ak_array,
                    allow_missing=True,
                ),
                dtype=np.float32,
            )
        )
    inputs = np.stack(variables_array, axis=-1)

    inputs_complete = {input_name[0]: inputs}

    outputs = session.run(output_name, inputs_complete)[0]
    return outputs
