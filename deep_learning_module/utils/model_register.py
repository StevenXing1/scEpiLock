from model.scEpiLock import scEpiLock
from model.scEpiLock_Hyena import scEpiLock_Hyena, scEpiLock_HyenaDeep
from utils.utils import Utils

class Model_Register():

    def __init__(self, model_name):
        self.model_name = model_name

    def get_model(self, n_class, seq_len=1000):

        if self.model_name == "scEpiLock":
            Utils.print_separator("scEpiLock")
            return scEpiLock(n_class)
        
        elif self.model_name == "scEpiLock_Hyena":
            Utils.print_separator("scEpiLock_Hyena")
            return scEpiLock_Hyena(n_class, seq_len=seq_len)
        
        elif self.model_name == "scEpiLock_HyenaDeep":
            Utils.print_separator("scEpiLock_HyenaDeep")
            return scEpiLock_HyenaDeep(n_class, seq_len=seq_len)

        else:
            Utils.print_separator("No Model Retrieved!")
            return
