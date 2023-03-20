from numpy import pi

class Geometry:
    def __init__(self):
        pass

    # general backbone geometry
    N_CA = 1.46
    CA_C = 1.53
    C_N = 1.33
    N_CA_C = 110 / 360 * (2 * pi)
    C_N_CA = 121 / 360 * (2 * pi)
    CA_C_N = 2.03

    # CB geometry
    CA_CB = 1.33
    N_CA_CB = 110 / 360 * (2 * pi)

    def C_N_CA_CB(self, phi):
        return phi + (237 / 360 * (2 * pi))
