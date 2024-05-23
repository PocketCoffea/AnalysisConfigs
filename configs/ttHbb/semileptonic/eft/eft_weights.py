import awkward as ak
import numpy as np
from pocket_coffea.lib.weights_manager import WeightCustom

#This function can be used only for the EFT center simulation, since in the SM center one you have one LHEReweightingweight
#less, since in the SMc simulationthe weight 0,0,..,0 is not calculated since that is already where the simulation is centered 



class EFTStructure:
    def __init__(self, reweight_card):
        self.reweight_card = reweight_card
        self.A_matrix = self.get_A_matrix()
        self.A_matrix_inv = self.invert_A_matrix()
        

    @classmethod
    def get_wilson_vector(cls, w_array):
        # warray starts with 1 for the SM
        w_array = [1] + w_array
        out = []
        for i in range(len(w_array)):
            for j in range(0, i+1):
                out.append(w_array[i]*w_array[j])
        return np.array(out)


    def get_A_matrix(self):
        with open(self.reweight_card) as f:
            lines = f.readlines()

        reweightings = [ ]
        Nparams = 8
        current_block = []
        inrw = False
        for iline in range(len(lines)):
            line = lines[iline]
            if not inrw:
                if line.startswith("launch"):
                    inrw = True
                    continue
            else:
                # reading a reweighting block
                current_block.append(float(line.split(" ")[-1]))
                if len(current_block) == Nparams:
                    # stop reading weight
                    inrw = False
                    reweightings.append(self.get_wilson_vector(current_block))
                    current_block.clear()

        return np.stack(reweightings)

    def invert_A_matrix(self):
        U, S, Vh = np.linalg.svd(self.A_matrix, full_matrices=True)
        ss = np.zeros((U.shape[0], S.shape[0]))
        np.fill_diagonal(ss, 1/S)
        return Vh.T @ ss.T @ U.T

    def get_structure_constants(self, weights):
        return (self.A_matrix_inv @ weights[:,:,None]).squeeze()


        
def getSMEFTweight(wilson_coeff):
    """
    Get the weight for the i-th SMEFT parameter point
    """
    name = f"SMEFT_weight_{'_'.join([str(x) for x in wilson_coeff])}"
    wilson_vector = EFTStructure.get_wilson_vector(wilson_coeff)
    return WeightCustom(
        name = name,
        function = lambda params, events,size, metadata, shape_variation: [
            (name, ak.to_numpy(events["EFT_struct"]) @ wilson_vector) ]
    )    


