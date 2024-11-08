#!/usr/bin/env python

# Program iq_dsp.py - Compute spectrum from I/Q data.
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
# 01-04-2014 Initial Release

import math, time
import numpy as np
import numpy.fft as fft

class DSP(object):
    def __init__(self, opt):
        self.opt = opt
        self.stats = list()
        # This is dB output for full scale 16bit input = max signal.
        self.db_adjust = 20. * math.log10(self.opt.size * 2**15)
        self.rejected_count = 0
        self.led_clip_ct = 0
        # Use "Hanning" window function
        self.w = np.empty(self.opt.size)
        for i in range(self.opt.size):
            self.w[i] = 0.5 * (1. - math.cos((2*math.pi*i)/(self.opt.size-1)))
        return

    def GetLogPowerSpectrum(self, data):
        size = self.opt.size
        #size = len(data) // self.opt.buffers
        power_spectrum = np.zeros(size)
        td_median = np.median(np.abs(data[:size])) if len(data[:size]) > 0 else 0
        td_threshold = (self.opt.pulse * td_median) * 2
        nbuf_taken = 0

        for ic in range(self.opt.buffers):
            td_segment = data[ic * size : (ic + 1) * size]
        
        
            if td_segment.size == 0:
                continue
        
            td_max = np.amax(np.abs(td_segment))
        
            if td_max < td_threshold:
            #if self.w.size != td_segment.size: #af mono
                #self.w = np.hanning(td_segment.size) #af mono
                td_segment *= self.w
                fd_spectrum = fft.fft(td_segment)
                fd_spectrum_rot = np.fft.fftshift(fd_spectrum)
                nbuf_taken += 1
                power_spectrum += np.real(fd_spectrum_rot * fd_spectrum_rot.conj())
            else:
                self.rejected_count += 1
                self.led_clip_ct = 1
    
        if nbuf_taken > 0:
            power_spectrum /= nbuf_taken
        else:
            power_spectrum = np.ones(size)
    
        power_spectrum = np.maximum(power_spectrum, 1e-10)
        log_power_spectrum = 10. * np.log10(power_spectrum) 
        #print("Log Power Spectrum sample:", log_power_spectrum[:10])
        return log_power_spectrum - self.db_adjust

