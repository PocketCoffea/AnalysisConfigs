import numpy as np
import awkward as ak
import sys

sys.path.append("../")
from utils.prediction_selection import extract_predictions


def get_pairing_information(session, input_name, output_name, events, max_num_jets):

    pt = np.array(
        np.log(
            ak.to_numpy(
                ak.fill_none(
                    ak.pad_none(events.JetGood.pt, max_num_jets, clip=True),
                    value=0,
                ),
                allow_missing=True,
            )
            + 1
        ),
        dtype=np.float32,
    )

    eta = np.array(
        ak.to_numpy(
            ak.fill_none(
                ak.pad_none(events.JetGood.eta, max_num_jets, clip=True),
                value=0,
            ),
            allow_missing=True,
        ),
        dtype=np.float32,
    )

    phi = np.array(
        ak.to_numpy(
            ak.fill_none(
                ak.pad_none(events.JetGood.phi, max_num_jets, clip=True),
                value=0,
            ),
            allow_missing=True,
        ),
        dtype=np.float32,
    )

    btag = np.array(
        ak.to_numpy(
            ak.fill_none(
                ak.pad_none(events.JetGood.btagPNetB, max_num_jets, clip=True),
                value=0,
            ),
            allow_missing=True,
        ),
        dtype=np.float32,
    )

    mask = np.array(
        ak.to_numpy(
            ak.fill_none(
                ak.pad_none(
                    ak.ones_like(events.JetGood.pt),
                    max_num_jets,
                    clip=True,
                ),
                value=0,
            ),
            allow_missing=True,
        ),
        dtype=np.bool_,
    )

    #inputs = np.stack((pt, eta, phi, btag), axis=-1)
    inputs = np.stack((pt, eta, phi), axis=-1)
    inputs_complete = {input_name[0]: inputs, input_name[1]: mask}

    outputs = session.run(output_name, inputs_complete)

    return outputs


def get_best_pairings(outputs):

    # extract the best jet assignment from
    # the predicted probabilities

    # NOTE: here the way this was implemented was changed
    assignment_probability = np.stack((outputs[0], outputs[1]), axis=0)
    # assignment_probability = [outputs[0], outputs[1]]

    # print("\nassignment_probability", assignment_probability)
    # swap axis
    predictions_best = np.swapaxes(extract_predictions(assignment_probability), 0, 1)
    # assignment_probability=np.array(assignment_probability)

    # get the probabilities of the best jet assignment

    # NOTE: here the way this was implemented was changed
    num_events = assignment_probability.shape[1]
    # num_events = len(assignment_probability[0])

    range_num_events = np.arange(num_events)
    best_pairing_probabilities = np.ndarray((2, num_events))
    for i in range(2):
        # print("best", i)
        best_pairing_probabilities[i] = assignment_probability[
            i,
            range_num_events,
            predictions_best[:, i, 0],
            predictions_best[:, i, 1],
        ]
    best_pairing_probabilities_sum = np.sum(best_pairing_probabilities, axis=0)
    # print("\nbest_pairing_probabilities_sum", best_pairing_probabilities_sum)

    # set to zero the probabilities of the best jet assignment, the symmetrization and the same jet assignment on the other target
    for j in range(2):
        for k in range(2):
            # print("set to zero", j, k)
            assignment_probability[
                j,
                range_num_events,
                predictions_best[:, j, k],
                predictions_best[:, j, 1 - k],
            ] = 0
            assignment_probability[
                1 - j,
                range_num_events,
                predictions_best[:, j, k],
                predictions_best[:, j, 1 - k],
            ] = 0

    # print("\nassignment_probability new", assignment_probability)
    # extract the second best jet assignment from
    # the predicted probabilities
    # swap axis
    predictions_second_best = np.swapaxes(
        extract_predictions(assignment_probability), 0, 1
    )

    # get the probabilities of the second best jet assignment
    second_best_pairing_probabilities = np.ndarray((2, num_events))
    for i in range(2):
        # print("second best", i)
        second_best_pairing_probabilities[i] = assignment_probability[
            i,
            range_num_events,
            predictions_second_best[:, i, 0],
            predictions_second_best[:, i, 1],
        ]
    second_best_pairing_probabilities_sum = np.sum(
        second_best_pairing_probabilities, axis=0
    )
    # print(
    #     "\nsecond_best_pairing_probabilities_sum",
    #     second_best_pairing_probabilities_sum,
    # )

    return (
        predictions_best,
        best_pairing_probabilities_sum,
        second_best_pairing_probabilities_sum,
    )
