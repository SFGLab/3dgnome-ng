from re import split as rsplit, search
import numpy as np
import os


def intersectQ(interval1, interval2):
    start1, end1 = interval1
    start2, end2 = interval2
    if start2 < end1 and start1 < end2:
        return True
    else:
        return False


def read_vcf(filename, region, sample_name=None, alt_allele=0):
    sv_list = []
    check_snp = False
    import vcf
    vcf_reader = vcf.Reader(open(filename, 'r'))

    for record in vcf_reader:
        # chr_id = "chr" + record.CHROM
        chr_id = record.CHROM
        start = record.POS
        if sample_name:
            gtype = record.genotype(sample_name)['GT']
            alleles = rsplit(r'[|/]', gtype)
            if len(alleles) == 1:
                allele = alleles[0]
            else:
                allele = alleles[alt_allele]
            if allele == '.':
                print("Warning: Missing allele. Variant {}:{} skipped.".format(
                    chr_id, start))
                continue
            else:
                allele = int(allele)
                if allele == 0:
                    continue
                else:
                    allele -= 1
        else:
            allele = 0
        alt = record.ALT[allele]
        sv_type = alt.type
        if sv_type == 'MNV':
            sv_type = 'INS'
            seq = alt.sequence[1:]
            start += 1
            end = start
        elif sv_type == 'SNV':
            seq = record.REF[1:]
            if seq:
                sv_type = 'CN0'
                start += 1
                end = start + len(seq)
                seq = None
            else:
                check_snp = True
                continue
        elif search(r'INS', sv_type):
            sv_type = 'INS'
            seq = None
            #Change of coordinates by mwlasnowolski
            # start += 1
            try:
                end = record.INFO['END']
            except KeyError:
                try:
                    sv_len = record.INFO['SVLEN'][allele]
                except KeyError:
                    end = start
                else:
                    end = start + sv_len
        else:
            try:
                end = record.INFO['END']
            except KeyError:
                try:
                    sv_len = record.INFO['SVLEN'][allele]
                except KeyError:
                    raise
                else:
                    end = start + sv_len
            seq = None
        sv_list.append([chr_id, start, end, sv_type, seq])
    if check_snp:
        print("Warning: SNPs are not handled. All SNPs skipped.")
    return sv_list


def merge_intervals(intervals):
    sorted_by_lower_bound = sorted(intervals, key=lambda x: x[0])
    merged = []
    for higher in sorted_by_lower_bound:
        if not merged:
            merged.append(higher)
        else:
            lower = merged[-1]
            if higher[0] < lower[1]:
                upper_bound = max(lower[1], higher[1])
                merged[-1] = [lower[0], upper_bound]
            else:
                merged.append(higher)
    return merged


def merge_svs(svs):
    sorted_by_lower_bound = sorted(svs, key=lambda x: x[0])
    merged = []
    skipped = []
    for higher in sorted_by_lower_bound:
        if not merged:
            merged.append(higher)
        else:
            lower = merged.pop()
            if higher[0] <= lower[1]:
                if lower[2] == 'INS':
                    merged = merged + [lower, higher]
                else:
                    upper_bound = max(lower[1], higher[1])
                    if higher[2] == lower[2]:
                        merged.append([lower[0], upper_bound, lower[2], None])
                    else:
                        skipped.append([lower[0], upper_bound])
            else:
                merged.append(lower)
                if skipped:
                    lower = skipped[-1]
                    if higher[0] <= lower[1]:
                        upper_bound = max(lower[1], higher[1])
                        skipped[-1] = [lower[0], upper_bound]
                    else:
                        merged.append(higher)
                else:
                    merged.append(higher)
    if skipped:
        print("Warning: Overlapping variants found and skipped.")
    return merged


def merge_anchors(anchor_list):
    chr_ids = set((anchor[0] for anchor in anchor_list))
    merged_all = []
    for chr_id in chr_ids:
        anchors = [x for x in anchor_list if x[0] == chr_id]
        sorted_by_lower_bound = sorted(anchors, key=lambda x: x[1])
        merged = []
        for higher in sorted_by_lower_bound:
            if not merged:
                merged.append(higher)
            else:
                lower = merged[-1]
                if higher[1] < lower[2]:
                    upper_bound = max(lower[2], higher[2])
                    if higher[3] == lower[3]:
                        orient = higher[3]
                    elif higher[3] == "N":
                        orient = lower[3]
                    elif lower[3] == "N":
                        orient = higher[3]
                    else:
                        orient = "N"
                    merged[-1] = [chr_id, lower[1], upper_bound, orient]
                else:
                    merged.append(higher)
        merged_all = merged_all + merged
    return merged_all


def my_int(string):
    try:
        return int(string)
    except ValueError:
        return string


def read_bed(filename):
    with open(filename) as f:
        lines = [list(map(my_int, line.rstrip('\n').split('\t'))) for line in f if
                 line.strip()]
    return lines


def add_suffix(filename, suffix):
    slash_split = filename.split('/')
    dirname = '/'.join(slash_split[:-1])
    dot_split = slash_split[-1].split('.')
    out_filename = '.'.join(dot_split[:-1]) + '_' + suffix + '.' + \
                   dot_split[-1]
    if dirname:
        return dirname + '/' + out_filename
    else:
        return out_filename


def save_init_file(fname, dir_name, segments_fname, anchors_fnames, clusters_fnames, singletons_fname,
                   singletons_inter_fname, centromere_fname):
    f = open(fname)
    text_lines = f.readlines()
    f.close()
    for i in range(len(text_lines)):
        key = text_lines[i].split('=')[0].strip()
        if key == 'data_dir':
            text_lines[i] = 'data_dir = %s/\n' % dir_name
        if key == 'anchors':
            text_lines[i] = 'anchors = %s\n' % anchors_fnames
        if key == 'clusters':
            text_lines[i] = 'clusters = %s\n' % clusters_fnames
        if key == 'segment_split':
            # text_lines[i] = 'segment_split = %s/%s\n' % (dir_name, segments_fname)
            text_lines[i] = 'segment_split = %s\n' % segments_fname
        if key == 'centromeres':
            text_lines[i] = 'centromeres = %s\n' % (centromere_fname)
        if key == 'singletons':
            text_lines[i] = 'singletons = %s\n' % singletons_fname
        if key == 'singletons_inter':
            text_lines[i] = 'singletons_inter = %s\n' % singletons_inter_fname

    out = ''.join(text_lines)
    outf = open(os.path.join(dir_name, 'stg.ini'), 'w')
    outf.write(out)
    outf.close()


def save_as_tsv(data, filename, sep='\t'):
    with open(filename, 'w') as outf:
        for line in data:
            out_line = sep.join(map(str, line))
            outf.write(out_line + '\n')


def save_as_tsv_specific_region(filename, region, output_name, path=None):
    from pybedtools import BedTool
    SVs_all = BedTool(filename)
    region_bed = BedTool(region.replace(':', '\t').replace('-', '\t'), from_string=True)
    SVs_in_region = SVs_all.intersect(region_bed, f=1)
    if not path:
        SVs_in_region.saveas(os.path.join(os.path.dirname(filename), output_name))
    else:
        SVs_in_region.saveas(os.path.join(path, output_name))



def percentiles(vals, ps):
    vals = np.array(vals)
    keys = ['P' + str(p) for p in ps]
    m = np.percentile(vals, ps)
    d = dict(list(zip(keys, m)))
    return d



