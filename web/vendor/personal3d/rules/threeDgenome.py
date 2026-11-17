from sortedcontainers import SortedList

from . import tools
from .motif import call_motifs
from re import search
import numpy as np


class Anchor(object):
    def __init__(self, chr_id, start, end, motif_orient):
        self.chr_id = chr_id
        self.start = start
        self.end = end
        self.motif_orient = motif_orient
        self._connected_anchors = {}

    def add_connected_anchor(self, anchor, weight):
        if anchor not in list(self._connected_anchors.keys()):
            self._connected_anchors[anchor] = weight
            anchor.add_connected_anchor(self, weight)

    def remove_connected_anchor(self, anchor):
        if anchor in list(self._connected_anchors.keys()):
            del self._connected_anchors[anchor]
            anchor.remove_connected_anchor(self)

    def get_connected_anchors(self):
        return self._connected_anchors.copy()

    def move_upstream(self, d):
        self.start -= d
        self.end -= d

    def move_downstream(self, d):
        self.start += d
        self.end += d

    def cut_left(self, pos):
        if self.start <= pos <= self.end:
            self.start = pos

    def cut_right(self, pos):
        if self.start <= pos <= self.end:
            self.end = pos

    def remove_contacts_upstream(self, pos):
        anchors = self.get_connected_anchors()
        for anchor in anchors:
            if anchor.end <= pos:
                self.remove_connected_anchor(anchor)

    def remove_contacts_downstream(self, pos):
        anchors = self.get_connected_anchors()
        for anchor in anchors:
            if anchor.start >= pos:
                self.remove_connected_anchor(anchor)

    def get_duplicate(self):
        duplicate = Anchor(self.chr_id, self.start, self.end,
                           self.motif_orient)
        for anchor, weight in list(self._connected_anchors.items()):
            duplicate.add_connected_anchor(anchor, weight)
        return duplicate

    def invert(self):
        d = {'R': 'L', 'L': 'R', 'N': 'N'}
        self.motif_orient = d[self.motif_orient]
        for connected_anchor in self.get_connected_anchors():
            self.remove_connected_anchor(connected_anchor)

    def inversion(self, interval):
        interval_start, interval_end = interval
        if tools.intersectQ((self.start, self.end), interval):
            self.invert()
            if not (self.start <= interval_start and self.end >= interval_end):
                end_d = interval_end - self.end
                start_d = interval_end - self.start
                self.start = interval_start + end_d
                self.end = interval_start + start_d

    def __lt__(self, other):
        return self.start < other.start

    def __repr__(self):
        return "Anchor({}, {}, {}, {})".format(self.chr_id, self.start,
                                               self.end, self.motif_orient)


class Segment(object):
    def __init__(self, chr_id, start, end):
        self.chr_id = chr_id
        self.start = start
        self.end = end
        self.neighbor_upstream = None
        self.neighbor_downstream = None
        self._anchors = SortedList([], key=lambda x: x.start)

    def set_neighbor_upstream(self, segment):
        if self.neighbor_upstream == segment:
            return
        neighbor_upstream_old = self.neighbor_upstream
        self.neighbor_upstream = segment
        if segment:
            if neighbor_upstream_old:
                neighbor_upstream_old.set_neighbor_downstream(None)
            segment.set_neighbor_downstream(self)

    def set_neighbor_downstream(self, segment):
        if self.neighbor_downstream == segment:
            return
        neighbor_downstream_old = self.neighbor_downstream
        self.neighbor_downstream = segment
        if segment:
            if neighbor_downstream_old:
                neighbor_downstream_old.set_neighbor_upstream(None)
            segment.set_neighbor_upstream(self)

    def add_anchor(self, anchor):
        self._anchors.add(anchor)

    def update_anchor_list(self, anchors):
        self._anchors = self._anchors + anchors

    def remove_anchor(self, anchor):
        connected_anchors = anchor.get_connected_anchors()
        for connected_anchor in list(connected_anchors.keys()):
            connected_anchor.remove_connected_anchor(anchor)
        self._anchors.remove(anchor)

    def get_anchor_list(self):
        return self._anchors.copy()

    def move_upstream(self, d):
        anchors = self.get_anchor_list()
        for anchor in anchors:
            self._anchors.remove(anchor)
            anchor.move_upstream(d)
            self._anchors.add(anchor)
        self.start -= d
        self.end -= d

    def move_downstream(self, d):
        anchors = self.get_anchor_list()
        for anchor in anchors:
            self._anchors.remove(anchor)
            anchor.move_downstream(d)
            self._anchors.add(anchor)
        self.start += d
        self.end += d

    def cut_left(self, pos):
        if self.start <= pos <= self.end:
            insert_ind = self._anchors.bisect_key_left(pos)
            for anchor in self._anchors[:insert_ind]:
                if anchor.end > pos:
                    self._anchors.remove(anchor)
                    anchor.cut_left(pos)
                    self._anchors.add(anchor)
                else:
                    self.remove_anchor(anchor)
            self.start = pos

    def cut_left_del(self, pos):
        if self.start <= pos <= self.end:
            insert_ind = self._anchors.bisect_key_left(pos)
            for anchor in self._anchors[:insert_ind]:
                self.remove_anchor(anchor)
            self.start = pos

    def cut_right(self, pos):
        if self.start <= pos <= self.end:
            insert_ind = self._anchors.bisect_key_left(pos)
            anchors = self.get_anchor_list()
            for anchor in anchors[insert_ind:]:
                self.remove_anchor(anchor)
            for anchor in self._anchors[:insert_ind]:
                if anchor.end > pos:
                    anchor.cut_right(pos)
            self.end = pos

    def cut_right_del(self, pos):
        if self.start <= pos <= self.end:
            insert_ind = self._anchors.bisect_key_left(pos)
            anchors = self.get_anchor_list()
            for anchor in anchors[insert_ind:]:
                self.remove_anchor(anchor)
            for anchor in self._anchors[:insert_ind]:
                if anchor.end > pos:
                    self.remove_anchor(anchor)
            self.end = pos

    def fuse_anchors(self, anchor1, anchor2):
        if not anchor1 and not anchor2:
            return None
        elif not anchor1:
            return anchor2
        elif not anchor2:
            return anchor1
        else:
            if anchor1.chr_id != anchor2.chr_id:
                raise ValueError
            else:
                chr_id = anchor1.chr_id
                fused_anchor = Anchor(chr_id,
                                      min(anchor1.start, anchor2.start),
                                      max(anchor1.end, anchor2.end),
                                      anchor1.motif_orient)
                for anchor in [anchor1, anchor2]:
                    for connected, weight in \
                            list(anchor.get_connected_anchors().items()):
                        fused_anchor.add_connected_anchor(connected, weight)
                    self.remove_anchor(anchor)
                self.add_anchor(fused_anchor)
                return fused_anchor

    def intersected_indices(self, interval):
        start, end = interval
        ind_last = self._anchors.bisect_key_left(end)
        potential_targets = self._anchors[:ind_last]
        ends = SortedList([anchor.end for anchor in potential_targets])
        ind_first = ends.bisect_right(start)
        return ind_first, ind_last

    def cut_out(self, start, end):
        if start > self.start and end < self.end:
            d = end - start
            ind_first, ind_last = self.intersected_indices((start, end))
            anchors = self.get_anchor_list()
            for anchor in anchors[ind_first:ind_last]:
                self.remove_anchor(anchor)
            for anchor in anchors[ind_last:]:
                self._anchors.remove(anchor)
                anchor.move_upstream(d)
                self._anchors.add(anchor)
            self.end = self.end - d
        else:
            raise ValueError

    def cut_in(self, start, end):
        if start > self.start and end < self.end:
            d = end - start
            ind_first, ind_last = self.intersected_indices((start, end))
            anchors = self.get_anchor_list()
            for anchor in anchors[ind_last:]:
                self._anchors.remove(anchor)
                anchor.move_downstream(d)
                self._anchors.add(anchor)
            affected_anchors = anchors[ind_first:ind_last]
            new_anchors = []
            for i in range(len(affected_anchors)):
                affected_anchor = affected_anchors[i]
                duplicate = affected_anchor.get_duplicate()
                affected_anchor.remove_contacts_downstream(end)
                duplicate.remove_contacts_upstream(start)
                for duplicate_connected in \
                        list(duplicate.get_connected_anchors().keys()):
                    if duplicate_connected in affected_anchors:
                        ind = affected_anchors.index(duplicate_connected)
                        if ind < i:
                            weight = duplicate.get_connected_anchors()[
                                duplicate_connected]
                            duplicate.add_connected_anchor(new_anchors[ind],
                                                           weight)
                        duplicate.remove_connected_anchor(duplicate_connected)
                duplicate.move_downstream(d)
                new_anchors.append(duplicate)
                self.add_anchor(duplicate)
            self.end = self.end + d
            self.update_connectivity()

    def get_duplicate(self):
        duplicate = Segment(self.chr_id, self.start, self.end)
        for i in range(len(self._anchors)):
            anchor = self._anchors[i]
            anchor_duplicate = Anchor(anchor.chr_id, anchor.start, anchor.end,
                                      anchor.motif_orient)
            duplicate.add_anchor(anchor_duplicate)
            connected_anchors = anchor.get_connected_anchors()
            for connected_anchor in list(connected_anchors.keys()):
                if connected_anchor in self._anchors:
                    ind = self._anchors.index(connected_anchor)
                    if ind < i:
                        anchor_duplicate.add_connected_anchor(
                            duplicate.get_anchor_list()[ind],
                            connected_anchors[connected_anchor])
                else:
                    anchor_duplicate.add_connected_anchor(
                        connected_anchor, connected_anchors[connected_anchor])
        return duplicate

    def remove_contacts_upstream(self, pos):
        for anchor in self._anchors:
            anchor.remove_contacts_upstream(pos)

    def remove_contacts_downstream(self, pos):
        for anchor in self._anchors:
            anchor.remove_contacts_downstream(pos)

    def find_closest_connection(self, anchor):
        if anchor.motif_orient == "L":
            ind = self._anchors.bisect_left(anchor)
            for elem in self._anchors[:ind]:
                d = (anchor.start + anchor.end)/2 - (elem.start + elem.end)/2
                if Genome.cluster_lengths['P10'] < d < \
                        Genome.cluster_lengths['P90']:
                    if elem.motif_orient == "R":
                        anchor.add_connected_anchor(elem, Genome.
                                                    cluster_weights_conv['P50']
                                                    )
                        break
                elif d >= Genome.cluster_lengths['P90']:
                    return
            if not anchor.get_connected_anchors() and self.neighbor_upstream:
                self.neighbor_upstream.find_closest_connection(anchor)
        elif anchor.motif_orient == "R":
            ind = self._anchors.bisect_right(anchor)
            for elem in self._anchors[ind:]:
                d = (elem.start + elem.end)/2 - (anchor.start + anchor.end)/2
                if Genome.cluster_lengths['P10'] < d < \
                        Genome.cluster_lengths['P90']:
                    if elem.motif_orient == "L":
                        anchor.add_connected_anchor(elem, Genome.
                                                    cluster_weights_conv['P50']
                                                    )
                        break
                elif d >= Genome.cluster_lengths['P90']:
                    return
            if not anchor.get_connected_anchors() and self.neighbor_downstream:
                self.neighbor_downstream.find_closest_connection(anchor)
        else:
            ind = self._anchors.bisect_right(anchor)
            for elem in self._anchors[ind:]:
                d = (elem.start + elem.end)/2 - (anchor.start + anchor.end)/2
                if Genome.cluster_lengths['P10'] < d < \
                        Genome.cluster_lengths['P90']:
                    anchor.add_connected_anchor(elem, Genome.cluster_weights[
                        'P50'])
                    break
                elif d >= Genome.cluster_lengths['P90']:
                    return
            if not anchor.get_connected_anchors() and self.neighbor_downstream:
                self.neighbor_downstream.find_closest_connection(anchor)

    def update_connectivity(self):
        for anchor in self._anchors:
            if not anchor.get_connected_anchors():
                self.find_closest_connection(anchor)

    def inversion(self, interval):
        interval_start, interval_end = interval
        if interval_start <= self.start < interval_end < self.end:
            self.cut_right(interval_end)
        elif self.start < interval_start < self.end <= interval_end:
            self.cut_left(interval_start)
        ind_first, ind_last = self.intersected_indices(interval)
        anchors = self.get_anchor_list()[ind_first:ind_last]
        for anchor in anchors:
            self._anchors.remove(anchor)
            anchor.inversion(interval)
            self._anchors.add(anchor)
        if not (self.start <= interval_start and self.end >= interval_end):
            end_d = interval_end - self.end
            start_d = interval_end - self.start
            self.start = interval_start + end_d
            self.end = interval_start + start_d
        self.update_connectivity()

    def insertion(self, pos, seq, pwm):
        seq_len = len(seq)
        self.end = self.end + seq_len
        ind_first, ind_last = self.intersected_indices((pos, pos+1))
        anchors = self.get_anchor_list()
        for anchor in anchors[ind_last:]:
            self._anchors.remove(anchor)
            anchor.move_downstream(seq_len)
            self._anchors.add(anchor)
        if ind_first != ind_last:
            affected_anchor = self._anchors[ind_first]
            duplicate = affected_anchor.get_duplicate()
            affected_anchor.cut_right(pos)
            affected_anchor.remove_contacts_downstream(pos)
            duplicate.cut_left(pos)
            duplicate.remove_contacts_upstream(pos)
            duplicate.move_downstream(seq_len)
            self.add_anchor(duplicate)
        motifs = call_motifs(seq, pwm)
        motif_len = pwm.shape[0]
        new_anchors = [Anchor(self.chr_id, pos+motif[0],
                              pos+motif[0]+motif_len, motif[1])
                       for motif in motifs]
        self.update_anchor_list(new_anchors)
        self.update_connectivity()

    def __lt__(self, other):
        return self.start < other.start

    def __repr__(self):
        return "Segment({}, {}, {})".format(self.chr_id, self.start, self.end)


class Genome(object):
    def __init__(self, segments, clusters, chr_lengths_dict, pwm=None):
        self.chr_lengths_dict = chr_lengths_dict
        self.pwm = pwm
        self.segments = dict([(chr_id, SortedList([], key=lambda x: x.start))
                              for chr_id in list(self.chr_lengths_dict.keys())])
        self.set_segments(segments)
        cluster_lengths, cluster_weights, cluster_weights_conv = \
            self.establish_connections(clusters)
        Genome.cluster_lengths = {key: int(value) for key, value in
                                  list(tools.percentiles(cluster_lengths,
                                                    [10, 50, 90]).items())
                                  }
        Genome.cluster_weights = {key: int(value) for key, value in
                                  list(tools.percentiles(cluster_weights,
                                                    [25, 50, 75]).items())
                                  }
        if cluster_weights_conv:
            Genome.cluster_weights_conv = {key: int(value) for key, value in
                                           list(tools.percentiles(
                                               cluster_weights_conv,
                                               [25, 50, 75]
                                           ).items())
                                           }

    def set_segments(self, segments):
        segment_dict = {}
        for segment in segments:
            chr_id = segment[0]
            if chr_id not in list(segment_dict.keys()):
                segment_dict[chr_id] = [segment[1:3]]
            else:
                segment_dict[chr_id].append(segment[1:3])
        for chr_id, segments in list(segment_dict.items()):
            merged_segments = tools.merge_intervals(segments)
            for i in range(len(merged_segments) - 1):
                seg_upstream_start, seg_upstream_end = merged_segments[i]
                seg_downstream_start, seg_downstream_end = merged_segments[i+1]
                self.segments[chr_id].add(Segment(chr_id, seg_upstream_start,
                                                  seg_upstream_end))
                if seg_upstream_end < seg_downstream_start:
                    seg_downstream = Segment(chr_id, seg_upstream_end,
                                             seg_downstream_start)
                    self.segments[chr_id].add(seg_downstream)
        for chr_id in list(self.segments.keys()):
            segment_list = self.segments[chr_id]
            if segment_list:
                first = segment_list[0]
                last = segment_list[-1]
                if first.start > 0:
                    self.segments[chr_id].add(Segment(chr_id, 0, first.start))
                if last.end < self.chr_lengths_dict[chr_id]:
                    self.segments[chr_id].add(
                        Segment(chr_id, last.end, self.chr_lengths_dict[chr_id]
                                ))
            else:
                self.segments[chr_id].add(Segment(chr_id, 0,
                                                  self.chr_lengths_dict[chr_id]
                                                  ))
        for chr_id in list(self.segments.keys()):
            segment_list = self.segments[chr_id]
            for i in range(len(segment_list) - 1):
                seg_upstream = segment_list[i]
                seg_downstream = segment_list[i + 1]
                seg_upstream.set_neighbor_downstream(seg_downstream)

    def establish_connections(self, clusters):
        clusters = (tuple(line) if line[1] <= line[4] else
                    tuple(line[3:6] + line[:3] + [line[7]] + [line[6], line[8]])
                    for line in clusters)
        clusters = set(clusters)
        clusters = list(map(list, clusters))
        for line in clusters:
            elems = [tuple(line[:3] + [line[6]]), tuple(line[3:6] + [line[7]])]
            weight = line[8]
            anchors = []
            segments = []
            for elem in elems:
                chr_id, start, end, motif_orient = elem
                if motif_orient not in ['R', 'L', 'N']:
                    raise ValueError("Invalid name for anchor orientation: "
                                     "{}. Supported names are: "
                                     "R - right, "
                                     "L - left, "
                                     "N - non-specified.".format(motif_orient))
                segment_list = self.segments[chr_id]
                ind = segment_list.bisect_key_right(start)
                ind = ind - 1
                if ind >= 0:
                    segment = segment_list[ind]
                    if segment.end >= end:
                        segments.append(segment)
                        anchor_list = segment.get_anchor_list()
                        ind = anchor_list.bisect_key_left(start)
                        if ind >= len(anchor_list):
                            anchor = Anchor(chr_id, start, end, motif_orient)
                            segment.add_anchor(anchor)
                        else:
                            while ind < len(anchor_list):
                                anchor = anchor_list[ind]
                                if anchor.start == start and \
                                        (anchor.end != end or
                                         anchor.motif_orient != motif_orient):
                                    ind = ind + 1
                                else:
                                    break
                            if anchor.start != start or anchor.end != end or \
                                    anchor.motif_orient != motif_orient:
                                anchor = Anchor(chr_id, start, end,
                                                motif_orient)
                                segment.add_anchor(anchor)
                        anchors.append(anchor)
                    else:
                        break
                else:
                    break
            if len(anchors) == 2:
                anchor_upstream, anchor_downstream = anchors
                anchor_upstream.add_connected_anchor(anchor_downstream, weight)
            elif len(anchors) == 1:
                if not anchors[0].get_connected_anchors():
                    segments[0].remove_anchor(anchors[0])

        cluster_lengths = [np.absolute((elem[1] + elem[2])/2 -
                                       (elem[4] + elem[5])/2)
                           for elem in clusters if elem[0] == elem[3]]
        cluster_weights = [elem[8] for elem in clusters]
        cluster_weights_conv = [elem[8] for elem in clusters
                                if (elem[1] <= elem[4] and elem[6] == 'R'
                                    and elem[7] == 'L')
                                or (elem[1] > elem[4] and elem[6] == 'L'
                                    and elem[7] == 'R')]

        return cluster_lengths, cluster_weights, cluster_weights_conv

    def add_segment(self, segment):
        chr_id = segment.chr_id
        if self.segments[chr_id]:
            insert_ind = self.segments[chr_id].bisect_right(segment)
            if insert_ind > 0:
                self.segments[chr_id][insert_ind - 1].set_neighbor_downstream(
                    segment)
            if insert_ind < len(self.segments):
                self.segments[chr_id][insert_ind].set_neighbor_upstream(
                    segment)
        self.segments[chr_id].add(segment)

    def remove_segment(self, segment):
        if segment.neighbor_upstream:
            segment.neighbor_upstream.set_neighbor_downstream(
                segment.neighbor_downstream)
        for anchor in segment.get_anchor_list():
            segment.remove_anchor(anchor)
        self.segments[segment.chr_id].remove(segment)

    def fuse_segments(self, segment1, segment2):
        if not segment1 and not segment2:
            return None
        elif not segment1:
            return segment2
        elif not segment2:
            return segment1
        else:
            if segment1.chr_id != segment2.chr_id:
                raise ValueError
            else:
                chr_id = segment1.chr_id
                fused_segment = Segment(chr_id,
                                        min(segment1.start, segment2.start),
                                        max(segment1.end, segment2.end))
                fused_segment.update_anchor_list(
                    segment1.get_anchor_list())
                fused_segment.update_anchor_list(
                    segment2.get_anchor_list())
                if segment1 in self.segments[chr_id]:
                    self.segments[chr_id].remove(segment1)
                if segment2 in self.segments[chr_id]:
                    self.segments[chr_id].remove(segment2)
                self.add_segment(fused_segment)
                return fused_segment

    def intersected_indices(self, genomic_interval):
        chr_id, start, end = genomic_interval
        segment_list = self.segments[chr_id]
        ind_last = segment_list.bisect_key_left(end)
        potential_targets = segment_list[:ind_last]
        ends = SortedList([segment.end for segment in potential_targets])
        ind_first = ends.bisect_right(start)
        return ind_first, ind_last

    def deletion(self, genomic_interval):
        chr_id, interval_start, interval_end = genomic_interval
        d = interval_end - interval_start
        ind_first, ind_last = self.intersected_indices(genomic_interval)
        for segment in self.segments[chr_id][ind_last:]:
            segment.move_upstream(d)
        affected_segments = self.segments[chr_id][ind_first:ind_last]
        right_cut = []
        neighbor_left = None
        neighbor_right = None
        for affected_segment in affected_segments:
            anchor_list = affected_segment.get_anchor_list()
            if anchor_list:
                anchor_first = anchor_list[0]
                anchor_last = anchor_list[-1]
                if anchor_first.start <= interval_start < anchor_first.end:
                    neighbor_left = affected_segment.neighbor_upstream
                if anchor_last.start < interval_end <= anchor_last.end:
                    neighbor_right = affected_segment.neighbor_downstream
            if affected_segment.start >= interval_start and \
                    affected_segment.end <= interval_end:
                self.remove_segment(affected_segment)
            elif affected_segment.start < interval_start and \
                    affected_segment.end > interval_end:
                affected_segment.cut_out(interval_start, interval_end)
            elif affected_segment.start < interval_start:
                affected_segment.cut_right_del(interval_start)
                right_cut.append(affected_segment)
            else:
                self.segments[chr_id].remove(affected_segment)
                affected_segment.cut_left_del(interval_end)
                affected_segment.move_upstream(d)
                self.segments[chr_id].add(affected_segment)
                if right_cut:
                    self.fuse_segments(right_cut.pop(), affected_segment)
        for i in range(ind_first, len(self.segments[chr_id])):
            segment = self.segments[chr_id].pop(i)
            self.segments[chr_id].add(segment)
        if neighbor_left:
            self.fuse_segments(neighbor_left,
                               neighbor_left.neighbor_downstream)
        if neighbor_right:
            self.fuse_segments(neighbor_right.neighbor_upstream,
                               neighbor_right)
        for segment in self.segments[chr_id]:
            segment.update_connectivity()
        self.chr_lengths_dict[chr_id] = self.chr_lengths_dict[chr_id] - d

    def duplication(self, genomic_interval):
        chr_id, interval_start, interval_end = genomic_interval
        d = interval_end - interval_start
        ind_first, ind_last = self.intersected_indices(genomic_interval)
        for segment in self.segments[chr_id][ind_last:]:
            self.segments[chr_id].remove(segment)
            segment.move_downstream(d)
            self.segments[chr_id].add(segment)
        affected_segments = self.segments[chr_id][ind_first:ind_last]
        duplicated_cut_left = []
        for affected_segment in affected_segments:
            if affected_segment.start >= interval_start and \
                    affected_segment.end <= interval_end:
                duplicate = affected_segment.get_duplicate()
                affected_segment.remove_contacts_downstream(interval_end)
                duplicate.remove_contacts_upstream(interval_start)
                duplicate.move_downstream(d)
                self.add_segment(duplicate)
            elif affected_segment.start < interval_start and \
                    affected_segment.end > interval_end:
                affected_segment.cut_in(interval_start, interval_end)
            elif affected_segment.start < interval_start:
                duplicate = affected_segment.get_duplicate()
                affected_segment.remove_contacts_downstream(interval_end)
                duplicate.cut_left(interval_start)
                duplicate.remove_contacts_upstream(interval_start)
                duplicate.move_downstream(d)
                self.add_segment(duplicate)
                duplicated_cut_left.append(duplicate)
            else:
                duplicate = affected_segment.get_duplicate()
                affected_segment.cut_right(interval_end)
                affected_segment.remove_contacts_downstream(interval_end)
                if duplicated_cut_left:
                    self.fuse_segments(affected_segment,
                                       duplicated_cut_left.pop())
                duplicate.remove_contacts_upstream(duplicate.start)
                duplicate.move_downstream(d)
                self.add_segment(duplicate)
        for segment in self.segments[chr_id]:
            segment.update_connectivity()
        self.chr_lengths_dict[chr_id] = self.chr_lengths_dict[chr_id] + d

    def inversion(self, genomic_interval):
        chr_id, interval_start, interval_end = genomic_interval
        ind_first, ind_last = self.intersected_indices(genomic_interval)
        affected_segments = self.segments[chr_id][ind_first:ind_last]
        inverted_cut_left = []
        cut_right = []
        neighbor_left = None
        neighbor_right = None
        if len(affected_segments) == 2:
            fused = self.fuse_segments(affected_segments[0],
                                       affected_segments[1])
            affected_segments = [fused]
        for affected_segment in affected_segments:
            anchor_list = affected_segment.get_anchor_list()
            if anchor_list:
                anchor_first = anchor_list[0]
                anchor_last = anchor_list[-1]
                if anchor_first.motif_orient == 'R' and \
                        anchor_last.motif_orient == 'L' and anchor_last in \
                        list(anchor_first.get_connected_anchors().keys()):
                    if anchor_first.start <= interval_start < anchor_first.end:
                        neighbor_left = affected_segment.neighbor_upstream
                    if anchor_last.start < interval_end <= anchor_last.end:
                        neighbor_right = affected_segment.neighbor_downstream
            if affected_segment.start < interval_start < affected_segment.end \
                    <= interval_end:
                duplicate = affected_segment.get_duplicate()
                affected_segment.cut_right(interval_start)
                cut_right.append(affected_segment)
                duplicate.inversion((interval_start, interval_end))
                self.add_segment(duplicate)
                inverted_cut_left.append(duplicate)
            elif interval_start <= affected_segment.start < interval_end < \
                    affected_segment.end:
                duplicate = affected_segment.get_duplicate()
                self.segments[chr_id].remove(affected_segment)
                affected_segment.cut_left(interval_end)
                self.segments[chr_id].add(affected_segment)
                duplicate.inversion((interval_start, interval_end))
                self.add_segment(duplicate)
                if cut_right:
                    self.fuse_segments(cut_right.pop(), duplicate)
                if inverted_cut_left:
                    self.fuse_segments(inverted_cut_left.pop(),
                                       affected_segment)
            else:
                self.segments[chr_id].remove(affected_segment)
                affected_segment.inversion((interval_start, interval_end))
                self.segments[chr_id].add(affected_segment)
        if neighbor_left:
            bordering_anchor = \
                neighbor_left.neighbor_downstream.get_anchor_list()[0]
            if bordering_anchor.motif_orient != 'R':
                self.fuse_segments(neighbor_left,
                                   neighbor_left.neighbor_downstream)
        if neighbor_right:
            bordering_anchor = \
                neighbor_right.neighbor_upstream.get_anchor_list()[-1]
            if bordering_anchor.motif_orient != 'L':
                self.fuse_segments(neighbor_right.neighbor_upstream,
                                   neighbor_right)
        for segment in self.segments[chr_id]:
            segment.update_connectivity()

    def insertion(self, chr_id, pos, seq):
        seq_len = len(seq)
        ind_first, ind_last = self.intersected_indices([chr_id, pos, pos+1])
        for segment in self.segments[chr_id][ind_last:]:
            self.segments[chr_id].remove(segment)
            segment.move_downstream(seq_len)
            self.segments[chr_id].add(segment)
        if self.pwm is not None:
            if ind_first == ind_last:
                motifs = sorted(call_motifs(seq, self.pwm), key=lambda x: x[0])
                orients = [motif[1] for motif in motifs]
                try:
                    left_motif_ind = orients.index('R')
                    orients = orients[left_motif_ind:]
                    right_motif_ind = list(reversed(orients)).index('L')
                    right_motif_ind = -right_motif_ind - 1
                except ValueError:
                    return
                else:
                    motifs = motifs[left_motif_ind:right_motif_ind+1]
                    motif_len = self.pwm.shape[0]
                    anchors = [Anchor(chr_id, pos+motif[0],
                                      pos+motif[0]+motif_len, motif[1])
                               for motif in motifs]
                    segment_start = anchors[0].start
                    segment_end = anchors[-1].end
                    new_segment = Segment(chr_id, segment_start, segment_end)
                    new_segment.update_anchor_list(anchors)
                    new_segment.update_connectivity()
                    self.add_segment(new_segment)
            else:
                affected_segment = self.segments[chr_id][ind_first]
                affected_segment.insertion(pos, seq, self.pwm)
            for segment in self.segments[chr_id]:
                segment.update_connectivity()
            self.chr_lengths_dict[chr_id] = self.chr_lengths_dict[chr_id] + \
                                            seq_len
        else:
            print("Warning: No Position Frequency Matrix provided. " \
                  "Insertion {}:{} skipped.".format(chr_id, pos))

    def introduce_sv(self, chr_id, start, end, sv_type, seq=None):

        match = search(r'([A-Za-z]+)(.*)', sv_type)
        sv_name = match.group(1)

        if sv_name == 'DUP':
            self.duplication([chr_id, start, end])
        elif sv_name == 'DEL':
            self.deletion([chr_id, start, end])
        # if sv_name == 'CN':
        #     copy_number = int(match.group(2))
        #     if copy_number == 0:
        #         self.deletion([chr_id, start, end])
        #     elif copy_number > 0:
        #         for i in range(copy_number - 1):
        #             self.duplication([chr_id, start, end])
        #     else:
        #         raise ValueError("Invalid copy number: {}.".format(
        #             copy_number))
        elif sv_name == 'INV':
            self.inversion([chr_id, start, end])
        elif sv_name == 'INS':
            if seq:
                self.insertion(chr_id, start, seq)
            else:
                print("Warning: No sequence specified. Insertion {}:{} " \
                      "skipped.".format(chr_id, start))
        else:
            raise ValueError("Unsupported SV type: {}.".format(sv_name))

        # match = search(r'([A-Za-z]+)(.*)', sv_type)
        # sv_name = match.group(1)
        # if sv_name == 'CN':
        #     copy_number = int(match.group(2))
        #     if copy_number == 0:
        #         self.deletion([chr_id, start, end])
        #     elif copy_number > 0:
        #         for i in range(copy_number - 1):
        #             self.duplication([chr_id, start, end])
        #     else:
        #         raise ValueError("Invalid copy number: {}.".format(
        #             copy_number))
        # elif sv_name == 'INV':
        #     self.inversion([chr_id, start, end])
        # elif sv_name == 'INS':
        #     if seq:
        #         self.insertion(chr_id, start, seq)
        #     else:
        #         print("Warning: No sequence specified. Insertion {}:{} " \
        #               "skipped.".format(chr_id, start))
        # else:
        #     raise ValueError("Unsupported SV type: {}.".format(sv_name))

    def map_svs_from_vcf(self, filename,  region, sample_name=None, alt_allele=0):
        sv_list = tools.read_vcf(filename, region, sample_name, alt_allele)
        sv_dict = {}
        merged_svs_genome = []

        region_filtered_sv_list = [i for i in sv_list if i[0] == region[0] and i[1] >= region[1] and i[2] <= region[2]]

        # for sv in sv_list:
        for sv in region_filtered_sv_list:
            chr_id = sv[0]
            if chr_id not in list(sv_dict.keys()):
                sv_dict[chr_id] = [sv[1:]]
            else:
                sv_dict[chr_id].append(sv[1:])
        for chr_id, svs in list(sv_dict.items()):
            merged_svs = tools.merge_svs(svs)
            for el in merged_svs:
                merged_svs_genome.append([chr_id]+el[:-1])

            for sv in reversed(merged_svs):
                start, end, sv_type, seq = sv
                self.introduce_sv(chr_id, start, end, sv_type, seq)

        return merged_svs_genome

    def get_segment_boundaries(self):
        boundaries = [[chr_id, segment.start, segment.end]
                      for chr_id in list(self.segments.keys())
                      for segment in self.segments[chr_id][1:]]
        return boundaries

    def get_interactions(self):
        interactions = ((anchor.chr_id,
                         anchor.start,
                         anchor.end,
                         connected_anchor.chr_id,
                         connected_anchor.start,
                         connected_anchor.end,
                         anchor.motif_orient,
                         connected_anchor.motif_orient,
                         weight)
                        if anchor.start <= connected_anchor.start else
                        (connected_anchor.chr_id,
                         connected_anchor.start,
                         connected_anchor.end,
                         anchor.chr_id,
                         anchor.start,
                         anchor.end,
                         connected_anchor.motif_orient,
                         anchor.motif_orient,
                         weight)
                        for chr_id in list(self.segments.keys())
                        for segment in self.segments[chr_id]
                        for anchor in segment.get_anchor_list()
                        for connected_anchor, weight in
                        list(anchor.get_connected_anchors().items()))
        interactions = set(interactions)
        interactions = list(map(list, interactions))
        return interactions
