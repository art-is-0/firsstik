import numpy as np 
from data_reader import read_xcom_table

class LaBr3Detector:
    def __init__(self):
        self.cmRadius = 3.5 * 2.54 / 2  # cm
        self.cmLength = 8.0 * 2.54  # cm
        self.g_cm3Density = 5.08  # g/cm^3
        self.cmDistanceToSource = 16.3 #cm

        self.xcomTables = read_xcom_table()  # dict of XCOM columns, see read_xcom_table
        self.xcomkeVEnergy = self.xcomTables["energyKeV"]


    def interpolatedLinearAttinuation(self,keVEnergy, process):
        """
        Interpolates XCOM mass attinuation in log-log space and converts it to linear attinuation.

        params:
            keVEnergy: photon energy or array of energies in keV
            process: key in self.xcomTables naming the XCOM column, e.g. "photoelectric_cm2_g"

        returns:
            interpolated linear attinuation, same shape as keVEnergy. 1/cm

        Assumptions:
            - Log-log interpolation, since XCOM coefficients are close to power laws between table points.
            - Linear interpolation below the first tabulated nonzero value. XCOM lists pair production
              as 0 at and below its thresholds (1022 keV nuclear, 2044 keV electron), so it stays 0 there
              and rises linearly up to the first nonzero point, without hard-coding the thresholds.

        Limitations:
            - Absorption-edge rows repeat an energy; np.interp does not treat these as exact jumps.
              The edges are below about 40 keV, so this does not matter for the peaks we simulate.

        LLM usage: Level 2 (Snippet); an LLM wrote the body of this method.
        """
        keVEnergy = np.asarray(keVEnergy, dtype=float)
        cm2_gAttinuation = self.xcomTables[process]

        isPositive = cm2_gAttinuation > 0  # log(0) is undefined; pair production is 0 below threshold
        keVTable = self.xcomkeVEnergy[isPositive]
        cm2_gTable = cm2_gAttinuation[isPositive]

        cm2_gInterpolated = np.exp(
            np.interp(np.log(keVEnergy), np.log(keVTable), np.log(cm2_gTable))
        )
        # Between a threshold (tabulated as 0) and the first nonzero point, log-log is undefined:
        # use linear interpolation there, which rises from 0 at the threshold.
        cm2_gLinear = np.interp(keVEnergy, self.xcomkeVEnergy, cm2_gAttinuation)
        cm2_gInterpolated = np.where(keVEnergy < keVTable[0], cm2_gLinear, cm2_gInterpolated)

        return cm2_gInterpolated * self.g_cm3Density

    def fwhm(self, keVEnergy): 
        """
        Returns the FWHM of the LaBr3 detector as a function of energy in keV. 
        Based upon OSCAR paper, with d=16.3 cm from target to detector"""
        a0 = 60.64
        a1 = 0.458
        a2 = 2.6555e-4
        resolution = np.sqrt(a0 + a1*keVEnergy + a2*keVEnergy**2)
        return resolution

    def efficiencyCurveEleniPS(self, keVEnergy): #point source 22cm 
        """
        Returns the efficiency of the LaBr3 detector as a function of energy in keV.
        Based upon Eleni's paper, with d=22 cm from target to detector"""
        efficiency = 2.874*keVEnergy**0.394/(7444 + keVEnergy**1.703)
        return efficiency


    def fractionFirstInteraction(self, counts, linearAttinuationTotal_1_cm_per_energy, randomGenerator):
        """ 
        Assumptions:
            - small angles , so photons travel in straight lines across the front face of the detector
            - point source on detectors's axis
            - ~5% Ce of the LaBr3(Ce) detectors is just treated as La
            - interactions not in crystal ignored

        Limitations: 
            - Only first interaction modeled
            - No coherent scattering since energy deposited is small, so can be neglected    
        """
        
        #what radius each photon passes into the crystal. assuming small angles based on detector geom ->uniform sampling
        cmRadiusPhotonFrontFace = self.cmRadius * np.sqrt(self.randomGenerator.random(counts))
        radiansAngle = 2* np.pi * self.randomGenerator.random(counts)

        #the photon can escape out through a side wall at a distribution of depths
        depthSidewallEscape = self.cmDistanceToSource * (self.cmRadius/ np.maximum(cmRadiusPhotonFrontFace, 1e-12) - 1)
        avaliableDepth = np.minimum(self.cmLength, depthSidewallEscape)

        uniformDepth = self.randomGenerator.random(counts)
        fractionInteracting = 1 - np.exp(-linearAttinuationTotal_1_cm_per_energy * avaliableDepth)
        cmDepth =  -np.log(1 - uniformDepth * fractionInteracting) / linearAttinuationTotal_1_cm_per_energy

        radiusInteractiondepth_cm = cmRadiusPhotonFrontFace * (self.cmDistanceToSource + cmDepth) / self.cmDistanceToSource
        cmX = radiusInteractiondepth_cm * np.cos(radiansAngle)
        cmY = radiusInteractiondepth_cm * np.sin(radiansAngle)

        return np.column_stack((cmX, cmY, cmDepth)), fractionInteracting