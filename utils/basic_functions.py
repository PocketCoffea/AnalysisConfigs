import awkward as ak


def add_fields(collection, fields=["pt", "eta", "phi", "mass"], four_vec=True):
    if four_vec:
        fields_dict = {field: getattr(collection, field) for field in fields}
        collection = ak.zip(
            fields_dict,
            with_name="PtEtaPhiMLorentzVector",
        )
    else:
        for field in fields:
            collection = ak.with_field(collection, getattr(collection, field), field)

    return collection
