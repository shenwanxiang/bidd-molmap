import time
from .util import get_data, normalize_index
from .psenacutil import extend_phyche_index, get_phyche_index, make_pseknc_vector, make_old_pseknc_vector


def check_psenac(lamada, w, k):
    """Check the validation of parameter lamada, w and k.
    """
    try:
        if not isinstance(lamada, int) or lamada <= 0:
            raise ValueError("Error, parameter lamada must be an int type and larger than and equal to 0.")
        elif w > 1 or w < 0:
            raise ValueError("Error, parameter w must be ranged from 0 to 1.")
        elif not isinstance(k, int) or k <= 0:
            raise ValueError("Error, parameter k must be an int type and larger than 0.")
    except ValueError:
        raise


def get_sequence_list_and_phyche_value_psednc(input_data, extra_phyche_index=None):
    """For PseDNC, PseKNC, make sequence_list and phyche_value.

    :param input_data: file type or handle.
    :param extra_phyche_index: dict, the key is the dinucleotide (string),
                                     the value is its physicochemical property value (list).
                               It means the user-defined physicochemical indices.
    """
    if extra_phyche_index is None:
        extra_phyche_index = {}

    original_phyche_value = {'AA': [0.06, 0.5, 0.27, 1.59, 0.11, -0.11],
                             'AC': [1.50, 0.50, 0.80, 0.13, 1.29, 1.04],
                             'AG': [0.78, 0.36, 0.09, 0.68, -0.24, -0.62],
                             'AT': [1.07, 0.22, 0.62, -1.02, 2.51, 1.17],
                             'CA': [-1.38, -1.36, -0.27, -0.86, -0.62, -1.25],
                             'CC': [0.06, 1.08, 0.09, 0.56, -0.82, 0.24],
                             'CG': [-1.66, -1.22, -0.44, -0.82, -0.29, -1.39],
                             'CT': [0.78, 0.36, 0.09, 0.68, -0.24, -0.62],
                             'GA': [-0.08, 0.5, 0.27, 0.13, -0.39, 0.71],
                             'GC': [-0.08, 0.22, 1.33, -0.35, 0.65, 1.59],
                             'GG': [0.06, 1.08, 0.09, 0.56, -0.82, 0.24],
                             'GT': [1.50, 0.50, 0.80, 0.13, 1.29, 1.04],
                             'TA': [-1.23, -2.37, -0.44, -2.24, -1.51, -1.39],
                             'TC': [-0.08, 0.5, 0.27, 0.13, -0.39, 0.71],
                             'TG': [-1.38, -1.36, -0.27, -0.86, -0.62, -1.25],
                             'TT': [0.06, 0.5, 0.27, 1.59, 0.11, -0.11]}

    sequence_list = get_data(input_data)
    phyche_value = extend_phyche_index(original_phyche_value, extra_phyche_index)

    return sequence_list, phyche_value


def get_sequence_list_and_phyche_value_pseknc(input_data, extra_phyche_index=None):
    """For PseDNC, PseKNC, make sequence_list and phyche_value.

    :param input_data: file type or handle.
    :param extra_phyche_index: dict, the key is the dinucleotide (string),
                                     the value is its physicochemical property value (list).
                               It means the user-defined physicochemical indices.
    """
    if extra_phyche_index is None:
        extra_phyche_index = {}

    original_phyche_value = {
        'AA': [0.06, 0.5, 0.09, 1.59, 0.11, -0.11],
        'AC': [1.5, 0.5, 1.19, 0.13, 1.29, 1.04],
        'GT': [1.5, 0.5, 1.19, 0.13, 1.29, 1.04],
        'AG': [0.78, 0.36, -0.28, 0.68, -0.24, -0.62],
        'CC': [0.06, 1.08, -0.28, 0.56, -0.82, 0.24],
        'CA': [-1.38, -1.36, -1.01, -0.86, -0.62, -1.25],
        'CG': [-1.66, -1.22, -1.38, -0.82, -0.29, -1.39],
        'TT': [0.06, 0.5, 0.09, 1.59, 0.11, -0.11],
        'GG': [0.06, 1.08, -0.28, 0.56, -0.82, 0.24],
        'GC': [-0.08, 0.22, 2.3, -0.35, 0.65, 1.59],
        'AT': [1.07, 0.22, 0.83, -1.02, 2.51, 1.17],
        'GA': [-0.08, 0.5, 0.09, 0.13, -0.39, 0.71],
        'TG': [-1.38, -1.36, -1.01, -0.86, -0.62, -1.25],
        'TA': [-1.23, -2.37, -1.38, -2.24, -1.51, -1.39],
        'TC': [-0.08, 0.5, 0.09, 0.13, -0.39, 0.71],
        'CT': [0.78, 0.36, -0.28, 0.68, -0.24, -0.62]}

    sequence_list = get_data(input_data)
    phyche_value = extend_phyche_index(original_phyche_value, extra_phyche_index)

    return sequence_list, phyche_value


def get_sequence_list_and_phyche_value(input_data, k, phyche_index, extra_phyche_index, all_property):
    """For PseKNC-general make sequence_list and phyche_value.

    :param input_data: file type or handle.
    :param k: int, the value of k-tuple.
    :param k: physicochemical properties list.
    :param extra_phyche_index: dict, the key is the dinucleotide (string),
                                     the value is its physicochemical property value (list).
                               It means the user-defined physicochemical indices.
    :param all_property: bool, choose all physicochemical properties or not.
    """
    if phyche_index is None:
        phyche_index = []
    if extra_phyche_index is None:
        extra_phyche_index = {}

    diphyche_list = ['Base stacking', 'Protein induced deformability', 'B-DNA twist', 'Dinucleotide GC Content',
                     'A-philicity', 'Propeller twist', 'Duplex stability:(freeenergy)',
                     'Duplex tability(disruptenergy)', 'DNA denaturation', 'Bending stiffness', 'Protein DNA twist',
                     'Stabilising energy of Z-DNA', 'Aida_BA_transition', 'Breslauer_dG', 'Breslauer_dH',
                     'Breslauer_dS', 'Electron_interaction', 'Hartman_trans_free_energy', 'Helix-Coil_transition',
                     'Ivanov_BA_transition', 'Lisser_BZ_transition', 'Polar_interaction', 'SantaLucia_dG',
                     'SantaLucia_dH', 'SantaLucia_dS', 'Sarai_flexibility', 'Stability', 'Stacking_energy',
                     'Sugimoto_dG', 'Sugimoto_dH', 'Sugimoto_dS', 'Watson-Crick_interaction', 'Twist', 'Tilt', 'Roll',
                     'Shift', 'Slide', 'Rise']
    triphyche_list = ['Dnase I', 'Bendability (DNAse)', 'Bendability (consensus)', 'Trinucleotide GC Content',
                      'Nucleosome positioning', 'Consensus_roll', 'Consensus-Rigid', 'Dnase I-Rigid', 'MW-Daltons',
                      'MW-kg', 'Nucleosome', 'Nucleosome-Rigid']

    # Set and check physicochemical properties.
    phyche_list = []
    if k == 2:
        phyche_list = diphyche_list
    elif k == 3:
        phyche_list = triphyche_list

    try:
        if all_property is True:
            phyche_index = phyche_list
        else:
            for e in phyche_index:
                if e not in phyche_list:
                    error_info = 'Sorry, the physicochemical properties ' + e + ' is not exit.'
                    raise NameError(error_info)
    except NameError:
        raise

    # Generate phyche_value and sequence_list.


    phyche_value = extend_phyche_index(get_phyche_index(k, phyche_index), extra_phyche_index)
    sequence_list = get_data(input_data)

    return sequence_list, phyche_value


class PseDNC():
    def __init__(self, lamada=3, w=0.05):
        self.lamada = lamada
        self.w = w
        self.k = 2
        check_psenac(self.lamada, self.w, self.k)

    def make_vec(self, input_data, extra_phyche_index=None):
        """Make PseDNC vector.

        :param input_data: file type or handle.
        :param extra_phyche_index: dict, the key is the dinucleotide (string),
                                         the value is its physicochemical property value (list).
                                   It means the user-defined physicochemical indices.
        """
        sequence_list, phyche_value = get_sequence_list_and_phyche_value_psednc(input_data, extra_phyche_index)


        vector = make_pseknc_vector(sequence_list, self.lamada, self.w, self.k, phyche_value, theta_type=1)

        return vector


class PseKNC():
    """This class should be used to make PseKNC vector."""

    def __init__(self, k=3, lamada=1, w=0.5):
        """
        :param k: k-tuple.
        """
        self.k = k
        self.lamada = lamada
        self.w = w
        check_psenac(self.lamada, self.w, self.k)

    def make_vec(self, input_data, extra_phyche_index=None):
        """Make PseKNC vector.

        :param input_data: file type or handle.
        :param extra_phyche_index: dict, the key is the dinucleotide (string),
                                         the value is its physicochemical property value (list).
                                   It means the user-defined physicochemical indices.
        """
        sequence_list, phyche_value = get_sequence_list_and_phyche_value_pseknc(input_data, extra_phyche_index)


        return make_old_pseknc_vector(sequence_list, self.lamada, self.w, self.k, phyche_value, theta_type=1)


class PCPseDNC():
    def __init__(self, lamada=1, w=0.05):
        self.lamada = lamada
        self.w = w
        self.k = 2
        check_psenac(self.lamada, self.w, self.k)

    def make_vec(self, input_data, phyche_index=None, all_property=False, extra_phyche_index=None):
        """Make a PCPseDNC vector.

        :param input_data: file object or sequence list.
        :param phyche_index: physicochemical properties list.
        :param all_property: choose all physicochemical properties or not.
        :param extra_phyche_index: dict, the key is the dinucleotide (string),
                                         the value is its physicochemical property value (list).
                                   It means the user-defined physicochemical indices.
        """
        # Make vector.
        sequence_list, phyche_value = get_sequence_list_and_phyche_value(input_data, self.k, phyche_index,
                                                                         extra_phyche_index, all_property)


        vector = make_pseknc_vector(sequence_list, self.lamada, self.w, self.k, phyche_value, theta_type=1)

        return vector


class PCPseTNC():
    def __init__(self, lamada=1, w=0.05):
        self.lamada = lamada
        self.w = w
        self.k = 3
        check_psenac(self.lamada, self.w, self.k)

    def make_vec(self, input_data, phyche_index=None, all_property=False, extra_phyche_index=None):
        """Make a PCPseDNC vector.

        :param input_data: file object or sequence list.
        :param phyche_index: physicochemical properties list.
        :param all_property: choose all physicochemical properties or not.
        :param extra_phyche_index: dict, the key is the dinucleotide (string),
                                         the value is its physicochemical property value (list).
                                   It means the user-defined physicochemical indices.
        """
        sequence_list, phyche_value = get_sequence_list_and_phyche_value(input_data, self.k, phyche_index,
                                                                         extra_phyche_index, all_property)
        # Make vector.


        vector = make_pseknc_vector(sequence_list, self.lamada, self.w, self.k, phyche_value, theta_type=1)

        return vector


class SCPseDNC():
    def __init__(self, lamada=1, w=0.05):
        self.lamada = lamada
        self.w = w
        self.k = 2
        check_psenac(self.lamada, self.w, self.k)

    def make_vec(self, input_data, phyche_index=None, all_property=False, extra_phyche_index=None):
        """Make a SCPseDNC vector.

        :param input_data: file object or sequence list.
        :param phyche_index: physicochemical properties list.
        :param all_property: choose all physicochemical properties or not.
        :param extra_phyche_index: dict, the key is the dinucleotide (string),
                                         the value is its physicochemical property value (list).
                                   It means the user-defined physicochemical indices.
        """
        sequence_list, phyche_value = get_sequence_list_and_phyche_value(input_data, self.k, phyche_index,
                                                                         extra_phyche_index, all_property)
        # Make vector.


        vector = make_pseknc_vector(sequence_list, self.lamada, self.w, self.k, phyche_value, theta_type=2)

        return vector


class SCPseTNC():
    def __init__(self, lamada=1, w=0.05):
        self.lamada = lamada
        self.w = w
        self.k = 3
        check_psenac(self.lamada, self.w, self.k)

    def make_vec(self, input_data, phyche_index=None, all_property=False, extra_phyche_index=None):
        """Make a SCPseTNC vector.

        :param input_data: file object or sequence list.
        :param phyche_index: physicochemical properties list.
        :param all_property: choose all physicochemical properties or not.
        :param extra_phyche_index: dict, the key is the dinucleotide (string),
                                         the value is its physicochemical property value (list).
                                   It means the user-defined physicochemical indices.
        """
        sequence_list, phyche_value = get_sequence_list_and_phyche_value(input_data, self.k, phyche_index,
                                                                         extra_phyche_index, all_property)
        # Make vector.


        vector = make_pseknc_vector(sequence_list, self.lamada, self.w, self.k, phyche_value, theta_type=2)

        return vector


if __name__ == '__main__':
    # psednc = PseDNC(lamada=2, w=0.05)
    # print(psednc.make_psednc_vec(['ACCCCA']))

    # PC_psednc = PCPseDNC(lamada=2, w=0.05)
    # print(PC_psednc.make_pcpsednc_vec(['ACCCCA'], phyche_index=["Tilt", 'Twist', 'Rise', 'Roll', 'Shift', 'Slide']))

    # pc_psetnc = PCPseTNC(lamada=2, w=0.05)
    # print(pc_psetnc.make_pcpsetnc_vec(['ACCCCA'], phyche_index=['Dnase I', 'Nucleosome']))

    # sc_psednc = SCPseDNC(lamada=2, w=0.05)
    # print(sc_psednc.make_scpsednc_vec(['ACCCCCA'], phyche_index=['Twist', 'Tilt']))

    # sc_psetnc = SCPseTNC(lamada=1, w=0.05)
    # print(sc_psetnc.make_scpsetnc_vec(['ACCCCCA'], phyche_index=['Dnase I', 'Nucleosome']))

    # sc_psetnc = SCPseTNC(lamada=2, w=0.05)
    # print(sc_psetnc.make_scpsetnc_vec(['ACCCCA'], phyche_index=["Dnase I", 'Nucleosome']))


    start_time = time.time()

    phyche_index = [[1.019, -0.918, 0.488, 0.567, 0.567, -0.070, -0.579, 0.488, -0.654, -2.455, -0.070, -0.918, 1.603,
                     -0.654, 0.567, 1.019]]

    print('Begin PseDNC')

    psednc = PseDNC()
    vec = psednc.make_vec(['GACTGAACTGCACTTTGGTTTCATATTATTTGCTC'])
    print(vec)
    print(len(vec[0]))

    psednc = PseDNC(lamada=2, w=0.1)
    vec = psednc.make_vec(['GACTGAACTGCACTTTGGTTTCATATTATTTGCTC'])
    print(vec)
    print(len(vec[0]))

    vec = psednc.make_vec(['GACTGAACTGCACTTTGGTTTCATATTATTTGCTC'],
                                 extra_phyche_index=normalize_index(phyche_index, is_convert_dict=True))
    print(vec)
    print(len(vec[0]))

    print('Begin PseKNC')

    pseknc = PseKNC()
    vec = pseknc.make_vec(['GACTGAACTGCACTTTGGTTTCATATTATTTGCTC'])
    print(vec)
    print(len(vec[0]))

    pseknc = PseKNC(k=2, lamada=1, w=0.05)
    vec = pseknc.make_vec(['GACTGAACTGCACTTTGGTTTCATATTATTTGCTC'])
    print(vec)
    print(len(vec[0]))

    vec = pseknc.make_vec(['GACTGAACTGCACTTTGGTTTCATATTATTTGCTC'],
                                 extra_phyche_index=normalize_index(phyche_index, is_convert_dict=True))
    print(vec)
    print(len(vec[0]))
    print()

    print('PC-PseDNC')
    pc_psednc = PCPseDNC()
    vec = pc_psednc.make_vec(['GACTGAACTGCACTTTGGTTTCATATTATTTGCTC'], phyche_index=['Twist', 'Tilt'])
    print(vec)
    print(len(vec[0]))

    pc_psednc = PCPseDNC(lamada=2, w=0.05)
    vec = pc_psednc.make_vec(['GACTGAACTGCACTTTGGTTTCATATTATTTGCTC'], all_property=True)
    print(vec)
    print(len(vec[0]))

    vec = pc_psednc.make_vec(['GACTGAACTGCACTTTGGTTTCATATTATTTGCTC'], phyche_index=['Twist', 'Tilt'],
                                      extra_phyche_index=normalize_index(phyche_index, is_convert_dict=True))
    print(vec)
    print(len(vec[0]))
    print()

    print('PC-PseTNC')
    pc_psetnc = PCPseTNC()
    vec = pc_psetnc.make_vec(['GACTGAACTGCACTTTGGTTTCATATTATTTGCTC'], phyche_index=['Dnase I', 'Nucleosome'])
    print(vec)
    print(len(vec[0]))

    pc_psetnc = PCPseTNC(lamada=2, w=0.05)
    vec = pc_psetnc.make_vec(['GACTGAACTGCACTTTGGTTTCATATTATTTGCTC'], all_property=True)
    print(vec)
    print(len(vec[0]))

    phyche_index = [
        [7.176, 6.272, 4.736, 7.237, 3.810, 4.156, 4.156, 6.033, 3.410, 3.524, 4.445, 6.033, 1.613, 5.087, 2.169, 7.237,
         3.581, 3.239, 1.668, 2.169, 6.813, 3.868, 5.440, 4.445, 3.810, 4.678, 5.440, 4.156, 2.673, 3.353, 1.668, 4.736,
         4.214, 3.925, 3.353, 5.087, 2.842, 2.448, 4.678, 3.524, 3.581, 2.448, 3.868, 4.156, 3.467, 3.925, 3.239, 6.272,
         2.955, 3.467, 2.673, 1.613, 1.447, 3.581, 3.810, 3.410, 1.447, 2.842, 6.813, 3.810, 2.955, 4.214, 3.581, 7.176]
    ]


    vec = pc_psetnc.make_vec(['GACTGAACTGCACTTTGGTTTCATATTATTTGCTC'], phyche_index=['Dnase I', 'Nucleosome'],
                                      extra_phyche_index=normalize_index(phyche_index, is_convert_dict=True))
    print(vec)
    print(len(vec[0]))
    print()

    print('SC-PseDNC')
    sc_psednc = SCPseDNC()
    vec = sc_psednc.make_vec(['GACTGAACTGCACTTTGGTTTCATATTATTTGCTC'], phyche_index=['Twist', 'Tilt'])
    print(vec)
    print(len(vec[0]))

    sc_psednc = SCPseDNC(lamada=2, w=0.05)
    vec = sc_psednc.make_vec(['GACTGAACTGCACTTTGGTTTCATATTATTTGCTC'], all_property=True)
    print(vec)
    print(len(vec[0]))

    phyche_index = [[1.019, -0.918, 0.488, 0.567, 0.567, -0.070, -0.579, 0.488, -0.654, -2.455, -0.070, -0.918, 1.603,
                     -0.654, 0.567, 1.019]]


    vec = sc_psednc.make_vec(['GACTGAACTGCACTTTGGTTTCATATTATTTGCTC'], phyche_index=['Twist', 'Tilt'],
                                      extra_phyche_index=normalize_index(phyche_index, is_convert_dict=True))
    print(vec)
    print(len(vec[0]))
    print()

    print('SC-PseTNC')
    sc_psetnc = SCPseTNC()
    vec = sc_psetnc.make_vec(['GACTGAACTGCACTTTGGTTTCATATTATTTGCTC'], phyche_index=['Dnase I', 'Nucleosome'])
    print(vec)
    print(len(vec[0]))

    sc_psetnc = SCPseTNC(lamada=2, w=0.05)
    vec = sc_psetnc.make_vec(['GACTGAACTGCACTTTGGTTTCATATTATTTGCTC'], all_property=True)
    print(vec)
    print(len(vec[0]))

    phyche_index = [
        [7.176, 6.272, 4.736, 7.237, 3.810, 4.156, 4.156, 6.033, 3.410, 3.524, 4.445, 6.033, 1.613, 5.087, 2.169, 7.237,
         3.581, 3.239, 1.668, 2.169, 6.813, 3.868, 5.440, 4.445, 3.810, 4.678, 5.440, 4.156, 2.673, 3.353, 1.668, 4.736,
         4.214, 3.925, 3.353, 5.087, 2.842, 2.448, 4.678, 3.524, 3.581, 2.448, 3.868, 4.156, 3.467, 3.925, 3.239, 6.272,
         2.955, 3.467, 2.673, 1.613, 1.447, 3.581, 3.810, 3.410, 1.447, 2.842, 6.813, 3.810, 2.955, 4.214, 3.581, 7.176]
    ]


    vec = sc_psetnc.make_vec(['GACTGAACTGCACTTTGGTTTCATATTATTTGCTC'], phyche_index=['Dnase I', 'Nucleosome'],
                                      extra_phyche_index=normalize_index(phyche_index, is_convert_dict=True))
    print(vec)
    print(len(vec[0]))

    # Normalize PseDNC index Twist, Tilt, Roll, Shift, Slide, Rise.
    original_phyche_value = [
        [0.026, 0.036, 0.031, 0.033, 0.016, 0.026, 0.014, 0.031, 0.025, 0.025, 0.026, 0.036, 0.017, 0.025, 0.016, 0.026],
        [0.038, 0.038, 0.037, 0.036, 0.025, 0.042, 0.026, 0.037, 0.038, 0.036, 0.042, 0.038, 0.018, 0.038, 0.025, 0.038],
        [0.020, 0.023, 0.019, 0.022, 0.017, 0.019, 0.016, 0.019, 0.020, 0.026, 0.019, 0.023, 0.016, 0.020, 0.017, 0.020],
        [1.69, 1.32, 1.46, 1.03, 1.07, 1.43, 1.08, 1.46, 1.32, 1.20, 1.43, 1.32, 0.72, 1.32, 1.07, 1.69],
        [2.26, 3.03, 2.03, 3.83, 1.78, 1.65, 2.00, 2.03, 1.93, 2.61, 1.65, 3.03, 1.20, 1.93, 1.78, 2.26],
        [7.65, 8.93, 7.08, 9.07, 6.38, 8.04, 6.23, 7.08, 8.56, 9.53, 8.04, 8.93, 6.23, 8.56, 6.38, 7.65]]
    for e in normalize_index(original_phyche_value, is_convert_dict=True).items():
        print(e)