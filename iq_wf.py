#!/usr/bin/env python

# Program iq_wf.py - Create waterfall spectrum display.
# Copyright (C) 2013-2014 Martin Ewing
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Contact the author by e-mail: aa6e@arrl.net
#
# Part of the iq.py program.

# HISTORY
# 01-04-2014 Initial release

import pygame as pg
import numpy as np
import math, sys

def palette_color(palette, val, vmin0, vmax0):
    """ translate a data value into a color according to several different
        methods. (PALETTE variable)
        input: value of data, minimum value, maximum value for transform
        return: pygame color tuple
    """
    f = (float(val) - vmin0) / (vmax0 - vmin0)     # btw 0 and 1.0
    f = min(1.0, max(0.0, f * 2))
    if palette == 1:
        r = int(255 * min(1.0, 3 * f))
        g = int(255 * min(1.0, 3 * (f - 0.333)))
        b = int(255 * min(1.0, 3 * (f - 0.666)))
    elif palette == 2:
        bright = min(1.0, f + 0.15)
        tpi_f = 2 * math.pi * f
        r = bright * 128 * (1.0 + math.cos(tpi_f))
        g = bright * 128 * (1.0 + math.cos(tpi_f + 2 * math.pi / 3))
        b = bright * 128 * (1.0 + math.cos(tpi_f + 4 * math.pi / 3))
    else:
        raise ValueError("Invalid palette requested!")
    return max(0, min(255, int(r))), max(0, min(255, int(g))), max(0, min(255, int(b)))

class Wf:
    """ Make a waterfall '3d' display of spectral power vs frequency & time.
        init: min, max palette parameter, no. of steps between min & max,
        size for each freq,time data plot 'pixel' (a box)
    """
    def __init__(self, opt, vmin, vmax, nsteps, pxsz):
        """ Initialize data and
            pre-calculate palette & filled rect surfaces, based on vmin, vmax,
            no. of surfaces = nsteps
        """
        self.opt = opt
        self.vmin, self.vmax = vmin, vmax
        self.vmin_rst, self.vmax_rst = vmin, vmax
        self.nsteps, self.pixel_size = nsteps, pxsz
        self.datasize, self.dx, self.firstcalc = None, None, True
        self.initialize_palette()
        
    def initialize_palette(self):
        """ Set up surfaces for each possible color value in list self.pixels.
        """
        self.pixels = []
        step_value = (self.vmax - self.vmin) / self.nsteps
        for istep in range(self.nsteps):
            color = palette_color(self.opt.waterfall_palette, self.vmin + istep * step_value, self.vmin, self.vmax)
            surface = pg.Surface(self.pixel_size)
            surface.fill(color)
            self.pixels.append(surface)

    def set_range(self, vmin, vmax):
        """ define a new data range for palette calculation going forward.
            input: vmin, vmax
        """
        self.vmin, self.vmax = vmin, vmax
        self.initialize_palette()

    def reset_range(self):
        """ reset palette data range to original settings.
        """
        self.vmin, self.vmax = self.vmin_rst, self.vmax_rst
        self.initialize_palette()
        return self.vmin, self.vmax

    def calculate(self, datalist, nsum, surface):   # (datalist is np.array)
        if self.firstcalc:
            self.datasize = len(datalist)
            self.wfacc = np.zeros(self.datasize)
            self.dx = surface.get_width() / self.datasize
            self.firstcalc = False

        self.wfacc += datalist
        self.wfcount = (self.wfcount + 1) if hasattr(self, 'wfcount') else 1

        if self.wfcount % nsum == 0:
            surface.scroll(dy=int(self.pixel_size[1]))
            normalized_data = np.clip((self.wfacc / nsum - self.vmin) / (self.vmax - self.vmin), 0, 1) * (self.nsteps - 1)
            for ix, val in enumerate(normalized_data.astype(int)):
                surface.blit(self.pixels[val], (int(ix * self.dx), 0))

            self.wfcount = 0
            self.wfacc.fill(0)
