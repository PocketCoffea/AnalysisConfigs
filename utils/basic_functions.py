import awkward as ak


def add_4vec_features(collection):
    collection=ak.with_field(collection, collection.pt, "pt")
    collection=ak.with_field(collection, collection.eta, "eta")
    collection=ak.with_field(collection, collection.phi, "phi")
    collection=ak.with_field(collection, collection.mass, "mass")

    return collection