# ✅ FSPM Radish Model + Assimilate Allocation + Light Environment Fractionation

from openalea.mtg import *
from gasexchange import GasExchange
from openalea.plantgl.all import *
import openalea.plantgl.all as pgl
import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator

# Leaf area interpolation function based on empirical data
df = pd.read_excel("leafarea_gpt.xlsx", sheet_name="leafarea")
points = df[["Temperature", "DAY", "Leaf Order"]].values
values = df["Leaf Area"].values
interp_func = LinearNDInterpolator(points, values)

def get_leaf_area_interp(temp: float, day: int, leaf_order: int) -> float:
    val = interp_func(temp, day, leaf_order)
    return float(val) if not np.isnan(val) else 0.0

# Parameters for leaf number increase
Rxleaf_germi  = 0.1679
Txleaf_germi  = 47.0948
Toleaf_germi  = 29.2426
Rxleaf  = 0.614856
Txleaf  = 54.65071
Toleaf  = 28.41466
conv = 1

class LeafNumber():
    def __init__(self, plantDensity):
        self.plantDensity = plantDensity
        self.leafNumber = 0.0
        self.germination = 0.0

    def calcLN(self, Ta):
        if ((Ta > 0.0) & (Ta < Txleaf_germi)):
            germRate = Rxleaf_germi * ((Txleaf_germi - Ta) / (Txleaf_germi - Toleaf_germi)) * (Ta / Toleaf_germi) ** (Toleaf_germi / (Txleaf_germi - Toleaf_germi))
            self.germination += germRate * conv

        if self.germination >= 1 and (Ta > 0.0) and (Ta < Txleaf):
            leafRate = Rxleaf * ((Txleaf - Ta) / (Txleaf - Toleaf)) * (Ta / Toleaf) ** (Toleaf / (Txleaf - Toleaf))
            self.leafNumber += leafRate * conv

# Growth model for assimilate partitioning
kgl = 0.031
kr = 0.015
ko = 0.020
Yg = 0.70
ggl = 1.463
gr = 1.444
go = 1.463
cd = 1.125

class Growth:
    def __init__(self):
        self.wgl = 1.0
        self.wr = 0.2
        self.wo = 0.0
        self.maint = 0.0

    def growCalc(self, Ta, assim, leafNumber, RDT=1.0):
        corr = 1 / 24
        assim_c = assim * RDT
        RM = (self.wgl * kgl + self.wr * kr + self.wo * ko) * 2 ** ((Ta - 20) / 10) * corr
        RMpr = min(RM, assim_c)
        fgl = 0.2323 + 0.6856 / (1 + np.exp(-(leafNumber + 25.5913) / -5.2098))
        fr = 1 - fgl
        fggl, fgr = fgl * ggl, fr * gr
        GT = fggl + fgr
        available = Yg * (assim_c - RMpr) / GT
        self.wgl += fgl * available
        self.wr  += fr  * available
        self.maint = assim_c - available

# Fractionation model (de Pury 1997)
class Fractionation():
    def __init__(self, latitude, press=101.3):
        self.lat = np.radians(latitude)
        self.P = press
        self.Icsun, self.Icsh = 0.0, 0.0
        self.laiSun, self.laiSh = 0.0, 0.0

    def radFraction(self, doy, hour, PPFD, LAI):
        a, fa, rhocd = 0.72, 0.426, 0.036
        rhoh, rhol, taul, sigma = 0.04, 0.10, 0.05, 0.15
        kd = 0.78 * 0.5
        kdprime = 0.719
        decl = -0.4093 * np.cos(2 * np.pi * (doy + 10) / 365)
        ha = np.pi / 12 * (hour - 12)
        incl = np.arccos(np.sin(decl)*np.sin(self.lat) + np.cos(decl)*np.cos(self.lat)*np.cos(ha))
        sunhgt = max(0.05, np.pi/2 - incl)
        kb = 0.5/np.sin(sunhgt) * 0.5
        kbprime = 0.46/np.sin(sunhgt)
        rhocb = 1 - np.exp(-2 * rhoh * kb / (1 + kb))
        m = self.P / 101.3 / np.sin(sunhgt)
        fd = (1 - a ** m) / (1 + a ** m * (1/fa - 1))
        It = PPFD
        Id = It * fd
        Ib = It - Id
        Icbs = Ib * (1 - rhocb) * (1 - np.exp(-kbprime * LAI))
        Icd = Id * (1 - rhocd) * (1 - np.exp(-kdprime * LAI))
        Icdb = Ib * (1 - sigma) * (1 - np.exp(-kb * LAI))
        Icdf = Id * (1 - rhocd) * (1 - np.exp(-(kdprime + kb) * LAI)) * kdprime / (kdprime + kb)
        Icsc = Ib * ((1 - rhocb) * (1 - np.exp(-(kbprime + kb) * LAI)) * kbprime / (kbprime + kb) - (1 - sigma) * (1 - np.exp(-2 * kb * LAI)) / 2)
        Icshdf = Id * (1 - rhocd) * ((1 - np.exp(-kdprime * LAI)) - (1 - np.exp(-(kdprime + kb) * LAI)) * kdprime / (kdprime + kb))
        Icshsc = Ib * ((1 - rhocb) * ((1 - np.exp(-kbprime * LAI)) - (1 - np.exp(-(kbprime + kb) * LAI)) * kbprime / (kbprime + kb)) - (1 - sigma) * ((1 - np.exp(-kb * LAI)) - (1 - np.exp(-2 * kb * LAI)) / 2))
        self.Icsun = Icdb + Icdf + Icsc
        self.Icsh = Icshdf + Icshsc

    def laiFraction(self, doy, hour, LAI):
        decl = -0.4093 * np.cos(2 * np.pi * (doy + 10) / 365)
        ha = np.pi / 12 * (hour - 12)
        incl = np.arccos(np.sin(decl)*np.sin(self.lat) + np.cos(decl)*np.cos(self.lat)*np.cos(ha))
        sunhgt = max(0.05, np.pi / 2 - incl)
        kb = 0.5 / np.sin(sunhgt) * 0.5
        self.laiSun = (1 - np.exp(-kb * LAI)) / kb
        self.laiSh = LAI - self.laiSun

# Simulation loop
plant_density = 7
leaf_model = LeafNumber(plantDensity=plant_density)
gmodel = Growth()
fmodel = Fractionation(latitude=35.0)

g = MTG()
plant = g.add_component(g.root)
root = g.add_child(plant, label='R')
leaf_area = g.property('area')
biomass = g.property('biomass')
assim = g.property('assim')
lai = g.property('lai')
geometry = g.property('geometry')

leaf_area[root] = biomass[root] = assim[root] = lai[root] = 0.0
geometry[root] = Sphere(radius=0.2)

prev_ln = 0
leaf_order_counter = 0
days = 6
for day in range(1, days + 1):
    Ta = 25.0
    doy = 120
    hour = 12

    leaf_model.calcLN(Ta)
    current_ln = int(leaf_model.leafNumber)

    if current_ln > prev_ln:
        for _ in range(current_ln - prev_ln):
            leaf_order_counter += 1
            node = g.add_child(root, edge_type='+', label='N')
            leaf = g.add_child(node, edge_type='/', label='L')

            area = get_leaf_area_interp(temp=Ta, day=day, leaf_order=leaf_order_counter)
            leaf_area[leaf] = area
            area_m2 = area / 10000
            lai[leaf] = area_m2 * plant_density

            radius = (area / 3.14) ** 0.5 / 100
            geometry[leaf] = Translated((0, leaf_order_counter * 0.03, 0), Sphere(radius=radius))

    prev_ln = current_ln

    LAI_total = sum([lai[v] for v in g.vertices(scale=1) if g.label(v) == 'L'])
    fmodel.radFraction(doy=doy, hour=hour, PPFD=300, LAI=LAI_total)
    fmodel.laiFraction(doy=doy, hour=hour, LAI=LAI_total)

    assimSun = fmodel.Icsun * fmodel.laiSun
    assimSh = fmodel.Icsh * fmodel.laiSh
    assimDay = (assimSun + assimSh) * conv / 1e6

    gmodel.growCalc(Ta, assimDay, current_ln)

assim[root] = assimDay
biomass[root] = gmodel.wr
lai[root] = LAI_total
geometry[root] = Sphere(radius=0.02 + biomass[root] / 1000.0)

print("[Assimilate Partitioning Result]")
print(f"Leaf biomass = {gmodel.wgl:.2f} g DM/m²")
print(f"Root biomass = {gmodel.wr:.2f} g DM/m²")
print(f"Respiration  = {gmodel.maint:.2f} g CH2O/m²")
print(f"Total LAI    = {lai[root]:.2f}")

scene = Scene([geometry[v] for v in g.vertices(scale=1) if v in geometry])
Viewer.display(scene)
