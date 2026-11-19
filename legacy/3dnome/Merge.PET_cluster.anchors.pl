#!/usr/bin/perl
use strict;
use warnings;
use Data::Dumper;
use File::Basename;
use List::Util qw ( max sum );

my $cluster_in = shift;
my $output     = shift;
my $ext        = 500;  # extened PET cluster anchor at both ends to merge anchors

print STDERR "Parsing PET cluster file:\n\t$cluster_in\n";
my $anchors_pos    = Parsing_PetCluster ( $cluster_in );

print STDERR "Connecting overlapped anchors ...\n";
my $extend_anchors = Anchor_Position_Extend ( $anchors_pos, $ext );

open ( my $fh_out, ">", $output ) or die $!;

for my $merged_anchor ( sort Sort_by_Anchor_Position keys %{ $extend_anchors } ){
    my ( $mergedAnchr_chr, $mergedAnchor_start, $mergedAnchor_end ) = split( /[:-]/, $merged_anchor );

    print $fh_out join( "\t", ($mergedAnchr_chr, $mergedAnchor_start, $mergedAnchor_end) ), "\n";
}

close $fh_out;


sub Anchor_Position_Extend {
    my ( $anchors_pos, $span ) = @_;

    my ( $chr, $start, $end, %extend_anchor, @connected_anchors, @PetCount );

    for my $anchor ( sort Sort_by_Anchor_Position keys %{ $anchors_pos } ) {

	my @anchor_infor = split(/[:-]/, $anchor);

	my $anchor_chr = $anchor_infor[0];
	my $anchor_s   = $anchor_infor[1];
	my $anchor_e   = $anchor_infor[2];
	my $anchor_PetCount = $anchors_pos->{$anchor};

	unless ( $chr && $start && $end && @connected_anchors ){
	    $chr   = $anchor_chr;
	    $start = $anchor_s;
	    $end   = $anchor_e;
	    push @connected_anchors, $anchor;
	    push @PetCount, $anchor_PetCount;
	    next;
	}


	if ( $chr eq $anchor_chr && ( $end + $span ) >= $anchor_s && $anchor_e >= $end ){
	    push @connected_anchors, $anchor;
	    push @PetCount, $anchor_PetCount;
	    $end = $anchor_e;

	} elsif ( $chr eq $anchor_chr && ( $end + $span ) >= $anchor_s && $anchor_e <= $end ){
	    push @connected_anchors, $anchor;
	    push @PetCount, $anchor_PetCount;
	
	} elsif ( $chr eq $anchor_chr && ( $end + $span ) < $anchor_s ){
	    $extend_anchor{$chr . ':' . $start . '-' . $end}{'anchors'}  = [ @connected_anchors ];
	    $extend_anchor{$chr . ':' . $start . '-' . $end}{'PetCount'} = [ @PetCount ];

	    $start = $anchor_s;
	    $end   = $anchor_e;
	    undef @connected_anchors;
	    undef @PetCount;
	    push @connected_anchors, $anchor;
	    push @PetCount, $anchor_PetCount;

	} elsif ( $chr ne $anchor_chr && $start && $end && @connected_anchors ){
	    $extend_anchor{$chr . ':' . $start . '-' . $end}{'anchors'}  = [ @connected_anchors ];
	    $extend_anchor{$chr . ':' . $start . '-' . $end}{'PetCount'} = [ @PetCount ];

	    $chr   = $anchor_chr;
	    $start = $anchor_s;
	    $end   = $anchor_e;
	    undef @connected_anchors;
	    undef @PetCount;
	    push @connected_anchors, $anchor;
	    push @PetCount, $anchor_PetCount;
	} else {
 	    print STDERR 'Unknown case: ', join("\t", ( $anchor_chr, $anchor_s, $anchor_e )), "\n";
	}
    }

    if ( @connected_anchors && @PetCount ){
	    $extend_anchor{$chr . ':' . $start . '-' . $end}{'anchors'}  = [ @connected_anchors ];
	    $extend_anchor{$chr . ':' . $start . '-' . $end}{'PetCount'} = [ @PetCount ];
	    undef @connected_anchors;
	    undef @PetCount;
	    undef $chr;
	    undef $start;
	    undef $end;
    }


    return \%extend_anchor;
}

sub Sort_by_Anchor_Position {
    my @a = split(/[:-]/, $a);
    my @b = split(/[:-]/, $b);

    $a[0] =~ /chr(.+)/;
    my $a_chr_index = $1;

    $b[0] =~ /chr(.+)/;
    my $b_chr_index = $1;

    if ( $a_chr_index eq 'X' or $a_chr_index eq 'x' ){
	push @a, 98;
    } elsif ( $a_chr_index eq 'Y' or $a_chr_index eq 'y' ){
	push @a, 99;
    } elsif ( $a_chr_index eq 'M' or $a_chr_index eq 'm' ){
	push @a, 100;
    } elsif ( $a_chr_index > 0 ){
	push @a, $a_chr_index;
    } else {
	print STDERR "Something wrong in chr index: $a_chr_index\n";
	exit 1;
    }

    if ( $b_chr_index eq 'X' or $b_chr_index eq 'x' ){
	push @b, 98;
    } elsif ( $b_chr_index eq 'Y' or $b_chr_index eq 'y' ){
	push @b, 99;
    } elsif ( $b_chr_index eq 'M' or $b_chr_index eq 'm' ){
	push @b, 100;
    } elsif ( $b_chr_index > 0 ){
	push @b, $b_chr_index;
    } else {
	print STDERR "Something wrong in chr index: $b_chr_index\n";
	exit 1;
    }

    $a[-1] <=> $b[-1]
	   ||
    $a[1] <=> $b[1]
           ||
    $a[2] <=> $b[2]
}


sub Parsing_PetCluster {
    my $in = shift;
    my %anchors;

    open(my $fh_in, $in) or die "Can't open input cluster file: $!";

    while (my $line = <$fh_in>){
	chomp $line;
	my @lines = split(/\t/, $line);

	if ( $lines[0] =~ /[C|c]hr[M|m]/ || $lines[3] =~ /[C|c]hr[M|m]/ ){
	    next;
	}

## head anchor position (chr1:3657531-3658374) and PET count
	$anchors{ $lines[0].':'.$lines[1].'-'.$lines[2] } = $lines[6];
## tail anchor position and PET count
	$anchors{ $lines[3].':'.$lines[4].'-'.$lines[5] } = $lines[6];
    }
    close $fh_in;
    return \%anchors;
}
