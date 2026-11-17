from pandas import read_csv
import numpy as np


def read_raw_pfm(filename):
    colnames = ['A', 'C', 'G', 'T']
    dataframe = read_csv(filename, sep=r'\s+', header=None, skiprows=1)
    dataframe = dataframe.T
    dataframe.columns = colnames
    return dataframe


def pfm_to_pwm(pfm):
    pfm = pfm + 1
    sums = pfm.sum(axis=1)
    ppm = pfm.div(sums, axis=0)
    pwm = np.log2(ppm * 4)
    return pwm


def sequence_score(seq, pwm):
    seq = seq.upper()
    score = 0
    for i in range(len(seq)):
        char = seq[i]
        score = score + pwm[char][i]
    return score


def calc_scores(seq, pwm):
    seq = seq.upper()
    compl_dict = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
    score_positive_strand = sequence_score(seq, pwm)
    seq_reversed = ''.join(reversed([compl_dict[char] for char in list(seq)]))
    score_negative_strand = sequence_score(seq_reversed, pwm)
    return score_positive_strand, score_negative_strand


def call_motifs(seq, pwm):
    seq_len = len(seq)
    motif_len = pwm.shape[0]
    if seq_len < motif_len:
        return []
    else:
        strongest_motifs = []
        steps_from_strongest = motif_len
        start = 0
        end = motif_len
        while end <= seq_len:
            score_positive_strand, score_negative_strand = calc_scores(
                seq[start:end], pwm)
            if score_positive_strand >= score_negative_strand:
                score = score_positive_strand
                orient = 'R'
            else:
                score = score_negative_strand
                orient = 'L'
            if score > 0:
                if steps_from_strongest < motif_len:
                    strongest = strongest_motifs.pop()
                    if score > strongest[2]:
                        strongest_motifs.append((start, orient, score))
                        steps_from_strongest = 1
                    else:
                        strongest_motifs.append(strongest)
                        steps_from_strongest += 1
                else:
                    strongest_motifs.append((start, orient, score))
                    steps_from_strongest = 1
            else:
                steps_from_strongest += 1
            start += 1
            end += 1
        return strongest_motifs
