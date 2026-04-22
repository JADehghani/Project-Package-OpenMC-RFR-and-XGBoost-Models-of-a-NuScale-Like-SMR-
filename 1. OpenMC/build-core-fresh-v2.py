#!/usr/bin/env python3
import argparse
from math import pi
from pathlib import Path

import numpy as np
import openmc
from tqdm import tqdm

from smr.materials import materials, clone
from smr.surfaces import lattice_pitch, bottom_fuel_stack, top_active_core, \
    pellet_OR, active_fuel_length
from smr.core import core_geometry
from smr import inlet_temperature


# Define command-line options
parser = argparse.ArgumentParser()
parser.add_argument('--multipole', action='store_true',
                    help='Use multipole cross sections')
parser.add_argument('--no-multipole', dest='multipole', action='store_false',
                    help='Do not use multipole cross sections')
parser.add_argument('--clone', action='store_true',
                    help='Clone materials for each cell instance')
parser.add_argument('--no-clone', dest='clone', action='store_false',
                    help='Do not clone materials for each cell instance')

# ------ Additional Simulation Parameters ------ #

# No. of annular rings in fuel
parser.add_argument('-r', '--rings', type=int, default=1, # Was 10, now 1 as rings are only neccessary for depletion calcs 
                    help='Number of annular regions in fuel')

# No. of axial subdivisions in fuel
parser.add_argument('-a', '--axial', type=int, default=5,# MIT suggests 196 (5 is reasonable to provide basic axial shape, matters for CRs)
                    help='Number of axial subdivisions in fuel')

# ---------------------------------------------- #

parser.add_argument('-d', '--depleted', action='store_true',
                    help='Whether UO2 compositions should represent depleted fuel')
parser.add_argument('-o', '--output-dir', type=Path, default=None)
parser.set_defaults(clone=False, multipole=True)
args = parser.parse_args()

####### ------------------------------------- INPUT INJECTION (Block 1) ---------------------------------- ######
# Updates Control rod positioning
# This block modifies the ZPlane surfaces that define Bank D control rod boundaries BEFORE geometry is built.
# Reads cr_step from inputs.txt and calculates the new rod position.
####### ---------------------------------------------------------------------------------------- ######  

# Make directory for inputs
if args.output_dir is None:
    if args.depleted:
        directory = Path('core-depleted')
    else:
        directory = Path('core-fresh')
else:
    directory = args.output_dir
directory.mkdir(exist_ok=True)

# Required Imports
import os
from openmc.data import atomic_weight # For b_ppm calculation
from smr.surfaces import surfs, bottom_fuel_stack # For control rod positioning


# Check if inputs.txt exists in the current directory
if os.path.exists("inputs.txt"):

    # ------ Read Input Parameters ------ #

    # Loads the input file created by generate_dataset.py
    with open("inputs.txt", "r") as f: 
        # Split the 6 input parameters between variables
        f_temp, c_temp, m_temp, m_dens, b_ppm, cr_step = map(float, f.read().split(',')) 

# ------ Control Rod Positioning ------ #
    # Modifies the ZPlane surfaces that define the boundaries of Bank D control rods.
    # Geometry:
    #   - Total rod travel: 380.635 cm (786.348 - 405.713)
    #   - Step size: 1.58173 cm/step
    #   - Range: 0 (fully inserted) to 248 (fully withdrawn)
    # Reference: surfaces.py, Uses math from Westinghouse BEAVRS benchmark

    # Dynamic SMR Step Calculation
    # Step 0 = Fully Inserted (Rod bottom sits at bottom of active fuel)
    # Step 248 = Fully Withdrawn (Rod bottom sits at top of active fuel)
    step_size = active_fuel_length / 248.0
    new_bot = bottom_fuel_stack + (cr_step * step_size)
    
    # Extract original rod length
    cr_length = surfs['bankD top'].z0 - surfs['bankD bot'].z0 
    new_top = new_bot + cr_length
    
    # Flatten the dashpot to prevent surface crossover errors in the reactor geometry
    surfs['dashpot top'].z0 = surfs['bottom FR'].z0
    
    # Update ZPlane surfaces for bank D (regulating rods) according to new rod boundaries
    surfs['bankD bot'].z0 = new_bot
    surfs['bankD top'].z0 = new_top 
    
######## ------------------ END OF INPUT INJECTION 1 ------------------------------------------ #########

if args.rings > 1:
    ring_radii = np.sqrt(np.arange(1, args.rings)*pellet_OR**2 / args.rings)
else:
    ring_radii = None
geometry = core_geometry(ring_radii, args.axial, args.depleted)

h = active_fuel_length / args.axial
fuel_mats = {}

# Count the number of instances for each cell and material
if args.clone:
    geometry.determine_paths(instances_only=True)

fuel_volume = pi * pellet_OR**2 * h / args.rings
for cell in tqdm(geometry.get_all_cells().values(),
                 desc='Differentiating materials / assigning volume'):
    if cell.fill in materials:
        # Determine if this material is fuel
        name = cell.fill.name
        is_fuel = 'UO2 Fuel' in name

        # Determine volume of each fuel material
        if is_fuel:
            if args.clone:
                # Fill cell with list of "differentiated" materials if requested
                cell.fill = [clone(cell.fill) for i in range(cell.num_instances)]
                for mat in cell.fill:
                    mat.volume = fuel_volume
            else:
                r_o = cell.region.bounding_box[1][0]
                if (name, r_o) not in fuel_mats:
                    cell.fill = cell.fill.clone()
                    cell.fill.volume = fuel_volume
                    fuel_mats[name, r_o] = cell.fill
                else:
                    cell.fill = fuel_mats[name, r_o]
        else:
            cell.fill.volume = 1.0

# Create OpenMC "materials.xml" file 
all_materials = geometry.get_all_materials()
materials = openmc.Materials(all_materials.values())

####### ------------------------------------- INPUT INJECTION (Block 2)---------------------------------- ######
# This block reads parameters from inputs.txt (written by generate_dataset.py) and injects them into 
# OpenMC material and geometry objects BEFORE they are exported to .xml in /core-fresh.
# Runs one n-batch simulation per sample.
#
# Parameters injected:
#   - 1) Fuel temperature - UO2 fuel materials
#   - 2) Cladding temperature - Zircaloy and M5 materials  
#   - 3) Moderator temperature - Borated water
#   - 4) Moderator density - Borated water
#   - 5) Boron concentration - Borated water composition
#   - 6) Control rod step - Bank D axial position (changes ZPlane surfaces)
####### ---------------------------------------------------------------------------------------- ######    
    # ------ Material Property Injection ------ #
 
 # Check if inputs.txt exists in the current directory
if os.path.exists("inputs.txt"):

    # ------ Read Input Parameters ------ #

    # Loads the input file created by generate_dataset.py
    with open("inputs.txt", "r") as f: 
        # Split the 6 input parameters between variables
        f_temp, c_temp, m_temp, m_dens, b_ppm, cr_step = map(float, f.read().split(','))
        
    # Loop through each material OpenMC just built in memory through this script
    for mat in materials:
        
        # 1) Update Fuel Temperature
        # Target: All UO2 fuel materials (1.6%, 2.4%, 3.1% enrichments)
        if 'UO2 Fuel' in mat.name:
            mat.temperature = f_temp
            
        # 2) Update Cladding Temperature
        # Target: Zircaloy-4 and M5 cladding alloys
        elif 'Zircaloy' in mat.name or 'M5' in mat.name:
            mat.temperature = c_temp
            
        # 3 & 4) Update Moderator Density and Boron concentration
        # Target: Borated water (coolant/moderator)
        elif 'Borated Water' in mat.name:
            mat.temperature = m_temp #sets moderator tempature
            mat.set_density('g/cc', m_dens) #sets corresonding moderator density
            
            # ------ Borated Water Recalculation (ppm to atomic fractions) ------ #
            # Concentration of boron ranges between 0-1200 ppm
            # Methodology from materials.py (ML17013A274, Figure 4.3-17)

            # Weight percent of natural boron in borated water
            wB_Bh2o = b_ppm * 1.0e-6

            # Borated water density
            rho_Bh2o = m_dens / (1 - wB_Bh2o)

            # Compute weight percent of clean water in borated water
            wh2o_Bh2o = 1.0 - wB_Bh2o

            # Compute molecular mass of clean water
            M_h2o = 2. * atomic_weight('H') + atomic_weight('O')

            # Compute molecular mass of borated water
            M_Bh2o = 1. / (wB_Bh2o / atomic_weight('B') + wh2o_Bh2o / M_h2o)

            # Compute atom fractions of boron and water
            aB_Bh2o = wB_Bh2o * M_Bh2o / atomic_weight('B')
            ah2o_Bh2o = wh2o_Bh2o * M_Bh2o / M_h2o

            # Compute atom fractions of hydrogen, oxygen
            ah_Bh2o = 2.0 * ah2o_Bh2o
            aho_Bh2o = ah2o_Bh2o

            # ------------------------------------------------------------ #

            # 5) Update Nuclides based on atom fractions calculated above (Boron Concetration)
            # Wipe the existing nuclide list and create new B10/B11 nuclides
            mat._nuclides.clear() # private attribute may break in future OpenMC versions
            mat.remove_element('B')
            mat.remove_element('H')
            mat.remove_element('O')
            mat.add_element('B', aB_Bh2o, 'ao')   
            mat.add_element('H', ah_Bh2o, 'ao')
            mat.add_element('O', aho_Bh2o, 'ao')    

######## ------------------ END OF INPUT INJECTION 2 ------------------------------------------ #########

materials.export_to_xml(str(directory / 'materials.xml'))


#### Create OpenMC "geometry.xml" file
geometry.export_to_xml(str(directory / 'geometry.xml'))


#### Create OpenMC "settings.xml" file

# Construct uniform initial source distribution over fissionable zones
lower_left = [-7.*lattice_pitch/2., -7.*lattice_pitch/2., bottom_fuel_stack]
upper_right = [+7.*lattice_pitch/2., +7.*lattice_pitch/2., top_active_core]
source = openmc.IndependentSource(space=openmc.stats.Box(lower_left, upper_right))
source.only_fissionable = True

settings = openmc.Settings()

# ------ Simulation Parameters ------ #
# Tuning justified in Section 3.2.6 of the final report

settings.batches = 200 # Batch no.
settings.inactive = 50 # Inactive batch no.
settings.particles = 10000 # No. of simulated particles

# ----------------------------------- #

settings.output = {'tallies': False}
settings.summary = False
settings.source = source
settings.sourcepoint = {'write': False}
settings.temperature = {
    'default': inlet_temperature,
    'method': 'interpolation',
    'range': (300.0, 1500.0),
}
if args.multipole:
    settings.temperature['multipole'] = True
    settings.temperature['tolerance'] = 1000

settings.export_to_xml(str(directory / 'settings.xml'))
