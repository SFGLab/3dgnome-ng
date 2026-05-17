/*
 * BedRegions.h
 *
 *  Created on: Jun 19, 2014
 *      Author: psz
 */

#ifndef BEDREGIONS_H_
#define BEDREGIONS_H_

#include <stdio.h>
#include <string>
#include <vector>

#include "BedRegion.h"

class BedRegions {
public:
	BedRegions();

	void fromFile(std::string path);

	void print();

	std::vector<BedRegion> regions;

};

#endif /* BEDREGIONS_H_ */
