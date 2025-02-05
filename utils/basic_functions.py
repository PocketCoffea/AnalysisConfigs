import awkward as ak


def add_fields(collection, fields=None, four_vec="PtEtaPhiMLorentzVector"):
    if fields==None:
        fields= list(collection.fields)
        for field in ["pt", "eta", "phi", "mass"]:
            if field not in fields:
                fields.append(field)
    if four_vec=="PtEtaPhiMLorentzVector":
        fields_dict = {field: getattr(collection, field) for field in fields}
        collection = ak.zip(
            fields_dict,
            with_name="PtEtaPhiMLorentzVector",
        )
    elif four_vec=="Momentum4D":
        fields=["pt", "eta", "phi", "mass"]
        fields_dict = {field: getattr(collection, field) for field in fields}
        collection = ak.zip(
            fields_dict,
            with_name="Momentum4D",
        )
    else:
        for field in fields:
            collection = ak.with_field(collection, getattr(collection, field), field)

    return collection
