from .nacutil import make_upto_kmer_list, make_revcomp_kmer_list, make_kmer_vector
from .util import get_data
from .nacutil import make_kmer_list, diversity, id_x_s

def check_nac_para(k, normalize=False, upto=False, alphabet='ACGT'):
    """Check the nac parameter's validation.
    """
    try:
        if not isinstance(k, int) or k <= 0:
            raise ValueError("Error, parameter k must be an integer and larger than 0.")
        elif not isinstance(normalize, bool):
            raise ValueError("Error, parameter normalize must be bool type.")
        elif not isinstance(upto, bool):
            raise ValueError("Error, parameter upto must be bool type.")
        elif alphabet != 'ACGT':
            raise ValueError("Error, parameter alphabet must be 'ACGT'.")
    except ValueError:
        raise


def get_kmer_list(k, upto, alphabet):
    """Get the kmer list.

    :param k: int, the k value of kmer, it should be larger than 0.
    :param upto: bool, whether to generate 1-kmer, 2-kmer, ..., k-mer.
    :param alphabet: string.
    """
    if upto:
        k_list = list(range(1, k + 1))
    else:
        k_list = list(range(k, k + 1))
    kmer_list = make_upto_kmer_list(k_list, alphabet)

    return kmer_list


class Kmer():
    def __init__(self, k=1, normalize=False, upto=False, alphabet="ACGT"):
        """
        :param k: int, the k value of kmer, it should be larger than 0.
        :param normalize: bool, normalize the result vector or not.
        :param upto: bool, whether to generate 1-kmer, 2-kmer, ..., k-mer.
        :param alphabet: string.
        """
        self.k = k
        self.upto = upto
        self.normalize = normalize
        self.alphabet = alphabet
        check_nac_para(k=self.k, upto=self.upto, normalize=self.normalize, alphabet=self.alphabet)
        self._kmer_list = get_kmer_list(self.k, self.upto, self.alphabet)
        self.feature_name_list = self._kmer_list
        
    def make_vec(self, data):
        """Make a kmer vector with options k, upto, revcomp, normalize.

        :param data: file object or sequence list.
        :return: kmer vector.
        """
        sequence_list = get_data(data)

        kmer_list = self._kmer_list

        rev_kmer_list = []
        revcomp = False
        vec = make_kmer_vector(sequence_list, kmer_list, rev_kmer_list, self.k, self.upto, revcomp, self.normalize)
        return vec


class RevcKmer():
    
    def __init__(self, k=1, normalize=False, upto=False, alphabet="ACGT"):
        """
        :param k: int, the k value of kmer, it should be larger than 0.
        :param normalize: bool, normalize the result vector or not.
        :param upto: bool, whether to generate 1-kmer, 2-kmer, ..., k-mer.
        :param alphabet: string.
        """
        self.k = k
        self.upto = upto
        self.normalize = normalize
        self.alphabet = alphabet
        check_nac_para(k=self.k, upto=self.upto, normalize=self.normalize, alphabet=self.alphabet)
        self._kmer_list = get_kmer_list(self.k, self.upto, self.alphabet)
        self._rev_kmer_list = make_revcomp_kmer_list(self._kmer_list)
        self.feature_name_list = ['r_%s' % i for i in self._rev_kmer_list]
        
    def make_vec(self, data):
        """Make a reverse compliment kmer vector with options k, upto, normalize.

        :param data: file object or sequence list.
        :return: reverse compliment kmer vector.
        """
        sequence_list = get_data(data)
        kmer_list = self._kmer_list
        # Use lexicographically first version of {kmer, revcomp(kmer)}.
        rev_kmer_list = self._rev_kmer_list
        revcomp = True
        vec = make_kmer_vector(sequence_list, kmer_list, rev_kmer_list, self.k, self.upto, revcomp, self.normalize)
        return vec


class IDkmer():
    def __init__(self, k=6, upto=True, alphabet='ACGT'):
        """
        :param k: int, the k value of kmer, it should be larger than 0.
        :param upto: bool, whether to generate 1-kmer, 2-kmer, ..., k-mer.
        :param alphabet: string.
        """
        self.k = k
        self.upto = upto
        self.alphabet = alphabet
        check_nac_para(k=self.k, upto=self.upto, alphabet=self.alphabet)
        feature_name_list = [('ID_pos_%s' % i, 'ID_neg_%s' % i) for i in range(1, k+1)]
        self.feature_name_list = [item for sublist in feature_name_list for item in sublist]
        
    def make_vec(self, data, hs, non_hs):
        """Make IDKmer vector.

        :param data: Need to processed FASTA file.
        :param hs: Positive FASTA file.
        :param non_hs: Negative FASTA file.
        """

        rev_kmer_list, upto, revcomp, normalize = [], False, False, False

        pos_s_list = get_data(hs)
        neg_s_list = get_data(non_hs)
        # print self.k
        if self.upto is False:
            k_list = [self.k]
        else:
            k_list = list(range(1, self.k+1))

        # print 'k_list =', k_list

        # Get all kmer ID from 1-kmer to 6-kmer.
        # Calculate standard source S vector.
        pos_s_vec, neg_s_vec = [], []
        diversity_pos_s, diversity_neg_s = [], []
        for k in k_list:
            kmer_list = make_kmer_list(k, self.alphabet)

            temp_pos_s_vec = make_kmer_vector(pos_s_list, kmer_list, rev_kmer_list, k, upto, revcomp, normalize)
            temp_neg_s_vec = make_kmer_vector(neg_s_list, kmer_list, rev_kmer_list, k, upto, revcomp, normalize)

            temp_pos_s_vec = [sum(e) for e in zip(*[e for e in temp_pos_s_vec])]
            temp_neg_s_vec = [sum(e) for e in zip(*[e for e in temp_neg_s_vec])]

            pos_s_vec.append(temp_pos_s_vec)
            neg_s_vec.append(temp_neg_s_vec)

            diversity_pos_s.append(diversity(temp_pos_s_vec))
            diversity_neg_s.append(diversity(temp_neg_s_vec))

        # Calculate Diversity(X) and ID(X, S).
        sequence_list = get_data(data)
        vec = []

        for seq in sequence_list:
            # print seq
            temp_vec = []
            for k in k_list:
                kmer_list = make_kmer_list(k, self.alphabet)
                seq_list = [seq]
                kmer_vec = make_kmer_vector(seq_list, kmer_list, rev_kmer_list, k, upto, revcomp, normalize)
                # print 'k', k
                # print 'kmer_vec', kmer_vec

                # print diversity_pos_s
                if upto is False:
                    k = 1

                # print 'pos_vec', pos_s_vec
                # print 'neg_vec', neg_s_vec
                # print 'diversity_pos_s', diversity_pos_s

                temp_vec.append(round(id_x_s(kmer_vec[0], pos_s_vec[k-1], diversity_pos_s[k-1]), 3))
                temp_vec.append(round(id_x_s(kmer_vec[0], neg_s_vec[k-1], diversity_neg_s[k-1]), 3))

            vec.append(temp_vec)

        return vec


if __name__ == '__main__':
    # kmer =Kmer(k=1)
    # kmer =RevcKmer(k=1, normalize=True, alphabet='ACGT')
    # kmer =IDkmer(k=1)


    kmer = Kmer(k=2)
    vec = kmer.make_vec(['GACTGAACTGCACTTTGGTTTCATATTATTTGCTC'])
    print("The vector is ", vec)

    kmer = Kmer(k=2, normalize=True)
    vec = kmer.make_vec(['GACTGAACTGCACTTTGGTTTCATATTATTTGCTC'])
    print("The vector is ", vec)

    kmer = Kmer(k=2, normalize=False, upto=True)
    vec = kmer.make_vec(['GACTGAACTGCACTTTGGTTTCATATTATTTGCTC'])
    print("The vector is ", vec)
    print('\n')



    revckmer = RevcKmer(k=2, normalize=False, upto=False)
    vec = revckmer.make_vec(['GACTGAACTGCACTTTGGTTTCATATTATTTGCTC'])
    print("The vector is ", vec)

    revckmer = RevcKmer(k=2, normalize=True, upto=False)
    vec = revckmer.make_vec(['GACTGAACTGCACTTTGGTTTCATATTATTTGCTC'])
    print("The vector is ", vec)

    revckmer = RevcKmer(k=2, normalize=True, upto=True)
    vec = revckmer.make_vec(['GACTGAACTGCACTTTGGTTTCATATTATTTGCTC'])
    print("The vector is ", vec)
    print('\n')

    print('Begin IDkmer.')


    print('Test: default mod.')
    idkmer = IDkmer()
    vec = idkmer.make_vec(open('test/example.fasta'), open('test/pos.fasta'), open('test/neg.fasta'))
    print(vec)

    print('Test: k=2.')
    idkmer = IDkmer(k=2)
    vec = idkmer.make_vec(open('test/example.fasta'), open('test/pos.fasta'), open('test/neg.fasta'))
    print(vec)

    print('Test: k=2, upto=False')
    idkmer = IDkmer(k=2, upto=False)
    vec = idkmer.make_vec(open('test/example.fasta'), open('test/pos.fasta'), open('test/neg.fasta'))
    print(vec)
    print('\n')

    # x = [11, 20, 13, 9, 27, 17, 1, 16, 9, 9, 14, 10, 6, 16, 13, 41]
    # s = [68, 67, 67, 58, 87, 52, 7, 76, 56, 46, 69, 48, 49, 58, 73, 90]
    # d_s = 3797.6619762268665
    # from nacutil import id_x_s
    # print id_x_s(x, s, d_s)