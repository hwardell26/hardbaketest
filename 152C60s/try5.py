import pySPM
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from pathlib import Path
import copy
import os
from IPython.display import display
import math as m
import scipy.optimize as opt
directory_in_str = str(os.getcwd())
directory = os.fsencode(directory_in_str)
#globals
lengthx = 1800 #<- length of line in nanometers
npillars = 2 #<- number of pillars to look for in scan
pixels = 512 #<- pixels in a horizontal line
period = 574.7 # <- period of grating in nm
periodpixellength = int(pixels/lengthx*period)
pixeltonmsf = lengthx/pixels
def removestreaks(profile):
    correctedprofile = profile.filter_scars_removal(.7,inline=False)
    return correctedprofile
def pillarlocator(profile, npillars, pillaraccuracy = 0.9): #finds first minimum and then adds approximate period length number of times of expected pillars, firstpillaraccuracy parameters can be adjusted between 0 and 1 to avoid measuring incompleted pillars
    scan1 = list(profile[0:int(pillaraccuracy*periodpixellength)])
    scan1min = min(scan1)
    scan1index = scan1.index(scan1min)
    trenches = []
    trenches.append(scan1index)
    i = 1
    while i < npillars+1:
        nextindex = scan1index + periodpixellength*i
        trenches.append(nextindex)
        i+=1
    return trenches
def peaklocator(profile, trenches): #pass output of pillar locator and overall profile to find maximum values within each region
    pillars = []
    i = 0
    while i < len(trenches)-1:
        pillar = max(profile[trenches[i]:trenches[i+1]])
        pillarindex = list(profile).index(pillar)
        pillars.append(pillarindex)
        i+=1
    return pillars
def trenchlocator(profile, peaks, pillaraccuracy = 0.9):
    print(f"trenchlocator: Received profile length: {len(profile)}")
    print(f"trenchlocator: Received peaks: {peaks}")
    if not peaks:# If peaks is empty, return an empty list.
        print("trenchlocator: 'peaks' list is empty. Returning empty trenches list.")
        return []
    trenches = []
    try:
        # Initial trench search:
        # Ensure the slice is not empty. If peaks[0] is 0, this slice will be empty.
        if peaks[0] > 0:
            trench = min(profile[0:peaks[0]])
            trenches.append(list(profile).index(trench))
        else:
            print("trenchlocator: First peak index is 0, cannot find trench before it.")
    except ValueError as e: # Catch specific error for min() on empty sequence
        print(f"trenchlocator: ValueError when finding initial trench: {e}")
        # If the first segment is empty might not find an initial trench.
        # This could mean the profile doesn't start with a trench, or data is bad.
        return [] # Return empty list if can't find the very first trench
    i = 0
    while i < len(peaks):
        try:
            # Determine the end of the search segment for the current trench
            if i + 1 < len(peaks):
                # Search between current peak and next peak
                search_end_index = peaks[i+1]
            else:
                # Last peak: search between current peak and an estimated 'period' length
                if npillars == 1:
                    previousperiod = periodpixellength # specific for mono pillar
                elif i > 0: # for multi-pillar, if it's the last peak
                    previousperiod = peaks[i] - peaks[i-1]
                else: # Fallback if first peak and no next peak (e.g., if npillars=1 and only one peak is detected)
                    previousperiod = periodpixellength # Default if no previous peak for estimation
                search_end_index = peaks[i] + int(previousperiod * pillaraccuracy)
            # Ensure the segment to search is valid and not empty
            segment_start = peaks[i]
            segment_end = min(search_end_index, len(profile)) # Don't go past profile end
            if segment_start < segment_end:
                trench = min(profile[segment_start:segment_end])
                trenches.append(list(profile).index(trench))
            else:
                print(f"trenchlocator: Skipping empty or invalid segment for trench {i}: [{segment_start}:{segment_end}]")

            i += 1
        except ValueError as e: # Catch specific error for min() on empty sequence
            print(f"trenchlocator: ValueError in loop for trench {i}: {e}. Skipping this trench.")
            i += 1 # Ensure i increments to avoid infinite loop
        except IndexError as e: # Catch if peaks[i-1] or peaks[i+1] is out of bounds
            print(f"trenchlocator: IndexError in loop for trench {i}: {e}. This might happen for last peak if no next peak exists.")
            i += 1
        except Exception as e: # Catch any other unexpected errors
            print(f"trenchlocator: An unexpected error occurred for trench {i}: {e}. Skipping this trench.")
            i += 1

    if len(trenches) > npillars + 1:
        # logic for finding trenches might be over-detecting
        print(f"trenchlocator: Warning! Found {len(trenches)} trenches, expected at most {npillars + 1}. Truncating.")
        # use the first expected number of trenches.
        trenches = trenches[:npillars + 1]

    return trenches # Always return a list
def trenchpillarcombiner(profile):
    peaks = peaklocator(profile, pillarlocator(profile, npillars))
    trenches = trenchlocator(profile, peaklocator(profile, pillarlocator(profile, npillars)))
    combinedlist = peaks + trenches
    combinedlist.sort()
    return combinedlist
def flatten(profile, npillars, flatline1delta = 0, flatline2delta = 0): #<- Flatten through lines drawn through each trench
    flatline1 = 0 + flatline1delta
    flatline2 = len(profile.pixels)-1-flatline2delta
    x1lines = trenchlocator(profile.pixels[flatline1], peaklocator(profile.pixels[flatline1], pillarlocator(profile.pixels[flatline1], npillars)))
    x2lines = trenchlocator(profile.pixels[flatline2], peaklocator(profile.pixels[flatline2], pillarlocator(profile.pixels[flatline2], npillars)))
    lines = []
    i = 0
    while i < len(x1lines):
        line = [x1lines[i],0,x2lines[i],len(profile.pixels)-1]
        lines.append(line)
        i+=1
    correctedprofile = profile.offset(lines)
    return correctedprofile
def fixzero1D(profile): #<- Pass 1D list
    minimum = min(profile)
    difference = 0 - minimum
    profile = list(map(lambda n: n + difference, profile))
    return profile
def fixzero2D(profile): #<- Pass 2D SPM Profile
    minimum = np.min(profile.pixels)
    with np.nditer(profile.pixels, op_flags=['readwrite']) as it:
        for x in it:
            x[...] = x - minimum
    return profile
def averageprofile(profile, outputerror = False): #function to take average profile of lines in an spm
    oneDprofile = np.flip(profile.mean(axis = 0))
    oneDerror = np.flip(profile.std(axis = 0))
    if outputerror:
        return (oneDprofile, oneDerror)
    else:
        return oneDprofile
def derivativeprofile(profile, n=3): #function that takes the average profile (one line) and returns the derivative, calculated n points away
    derivativelist = []
    # The loop should iterate from n up to len(profile) - n - 1 (inclusive)
    # to ensure that profile[i-n] and profile[i+n] are always valid.
    # So, the range should be [n, len(profile) - n)
    for i in range(n, len(profile) - n):
        # Use 'n' from the parameter, not a hardcoded '3'
        derivative = (profile[i+n] - profile[i-n])/(n*2*pixeltonmsf)
        derivativelist.append(derivative)
    return derivativelist
def pillarwidthcalc(profile, startingindex, endingindex, height = 0.10): #Calculates pillar with at height input, default at 10% height, always goes one pixel to right
    widthmeasurepoint = (height * (max(profile[startingindex:endingindex])-(0.5*(profile[startingindex]+profile[endingindex])))) + min(profile[startingindex:endingindex])
    peakcenterindex = profile[startingindex:endingindex].index(max(profile[startingindex:endingindex]))
    firstwall = profile[startingindex:endingindex][:peakcenterindex]
    secondwall = profile[startingindex:endingindex][peakcenterindex:]
    insertfirstwall = firstwall + [widthmeasurepoint]
    insertsecondwall = secondwall + [widthmeasurepoint]
    insertfirstwall.sort()
    insertsecondwall.sort()
    insertsecondwall.reverse()
    firstwallheightindex = insertfirstwall.index(widthmeasurepoint) - 1
    secondwallheightindex = insertsecondwall.index(widthmeasurepoint) - 1
    pillarwidth = (secondwallheightindex+len(firstwall)-firstwallheightindex)*pixeltonmsf
    return pillarwidth 
"""
def calculate_rq(profile_segment):#Calculates the Rq (root-mean-square) roughness value for a given profile segment.
#Args:
#profile_segment (list or numpy.ndarray): A 1D array or list representing the height profile.
#Returns:
#float: The Rq roughness value.
    if len(profile_segment) == 0:
        return 0.0  
    profile_segment = np.array(profile_segment)
    mean_height = np.mean(profile_segment)
    rq = np.sqrt(np.mean((profile_segment - mean_height)**2))
    return rq 
"""
def calculate_rq_middle_90_percent(profile_segment):
#Calculates the Rq (root-mean-square) roughness value for the middle 90%
#of a given profile segment based on its height range.
#Args:
#profile_segment (list or numpy.ndarray): A 1D array or list representing the height profile.
#Returns:
#float: The Rq roughness of value for the middle 90% height segment
#return 0.0 is segment is empty or too small to calculate percentile
    if len(profile_segment) == 0:
        return 0.0
    profile_segment=np.array(profile_segment)
#determine 5th and 95th percentiles of height data
    lower_bound_height = np.percentile(profile_segment, 5)
    upper_bound_height = np.percentile(profile_segment, 95)
#adjust segment to include only points within 5th and 95th percentiles
    middle_90_percent_segment = profile_segment[(profile_segment >= lower_bound_height) & (profile_segment <= upper_bound_height)]
    mean_height = np.mean(middle_90_percent_segment)
    rq = np.sqrt(np.mean((middle_90_percent_segment - mean_height)**2))
    return rq

if __name__ == "__main__":
    siteindex = 0
    siteindexes = []  
    maxpillarheights = [] #height outputs
    pillarheights = []
    pillardvangles = []#derivative angle outputs
    pillarangles = [] #wall angle calculated at 10% to 90%
    pillarwidths = []#pillarwidth
    dutycycle = []
    pillar_rq_values = []#Rq roughness
    i = 0
    while i<npillars*2:
        pillarheights.append([])
        pillardvangles.append([])
        pillarangles.append([])
        pillar_rq_values.append([]) # Initialize list for Rq values
        if i % 2 == 0:
            pillarwidths.append([])
        if i % 2 == 0:
            dutycycle.append([])
        i += 1
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".spm"):
            print(f"processing file:{filename}")
            Scan = pySPM.Bruker(filename)  
            topo = Scan.get_channel()
            topoE = copy.deepcopy(topo) #doesn't modify original file, remove to make edits permanent
            topoE = removestreaks(topoE)#Section to modify 2D Data
            topoE = flatten(topoE, npillars)
            topoE = fixzero2D(topoE)
        
            averageprofileoutput = averageprofile(topoE.pixels, outputerror = True)#Section to modify average profile
            averageprofilelist = fixzero1D(averageprofileoutput[0])
            derivativeprofilelist = derivativeprofile(averageprofilelist)
            """
            #Section to plot 2D profile
            fig,ax = plt.subplots()
            ax.imshow(topoE.pixels)
            plt.savefig(f'{filename.split('.')[0]} {filename.split('.')[1]}.png')
            mpl.pyplot.close()
        
            #Section to plot average profile
            x = np.linspace(0,len(topoE.pixels[0]),len(topoE.pixels[0]))
            y = list(averageprofilelist)
            fig,ax  = plt.subplots()
            ax.plot(x, y, linewidth=2.0)
            plt.savefig(f'{filename.split('.')[0]} {filename.split('.')[1]} Average Profile.png')
            mpl.pyplot.close()
            
            #Section to write average profile to excel
            df = pd.DataFrame(
                {'Average Height (nm)': averageprofilelist, 'standard deviation': averageprofileoutput[1]})
            writer = pd.ExcelWriter(f"{filename}.xlsx", engine='xlsxwriter')
            df.to_excel(writer, sheet_name= "average profile", index=False)
            writer._save()
            """
            df_avg_profile = pd.DataFrame(
                {'Average Height (nm)': averageprofilelist, 'standard deviation': averageprofileoutput[1]})
            try:
                with pd.ExcelWriter(f"{filename}.xlsx", engine='xlsxwriter') as writer:
                    df_avg_profile.to_excel(writer, sheet_name= "average profile", index=False)
                print(f"Average profile for '{filename}' written to '{filename}.xlsx'")
            except Exception as e:
                print(f"Error writing average profile for '{filename}': {e}")

            l = list(averageprofilelist)#Section to calculate important quantities
            d = list(map(lambda n: abs(n), derivativeprofilelist))
            maxpillarheights.append(max(l) - min(l))
            importantpoints = trenchpillarcombiner(l)
            p = importantpoints
            
            i = 0
            while i < npillars*2:
                pillarheights[i].append(abs(l[p[i+1]]-l[p[i]]))
                # pillardvangles[i].append(57.2958*m.atan(max(d[p[i]:p[i+1]])))
                # pillarangles[i].append(wallanglecalc(l,p[i],p[i+1]))
                
                # Calculate Rq for the wall segment
                # Assuming p[i] to p[i+1] represents a wall segment (trench to peak or peak to trench)
                wall_segment = l[p[i]:p[i+1]]
                pillar_rq_values[i].append(calculate_rq_middle_90_percent(wall_segment))
                
                if i % 2 == 0:
                    pillarwidths[int(i/2)].append(pillarwidthcalc(l, p[i], p[i+2]))
                if i % 2 == 0:
                    dutycycle[int(i/2)].append((pillarwidthcalc(l, p[i], p[i+2], height = 0.5))/period)
                i += 1

            siteindexes.append(siteindex)
            siteindex += 1
    foldername = str(os.getcwd()).split('\\')[-1]
    df = pd.DataFrame({"Sample Index": siteindexes})
    #I WANT TO REPLACE SAMPLE INDEX W/FILEPATH
    
    df['Max Pillar Height (nm)'] = maxpillarheights  
    i = 0
    while i < npillars*2:
        if i % 2 == 0:
            side = 'Left'
            pillarn = int(i/2 + 1)
        else:
            side = 'Right'
            pillarn = int((i-1)/2+1)
        df[f'Pillar {pillarn} {side} Height'] = pillarheights[i]
        i += 1
    i = 0# Add Rq values to the DataFrame
    while i < npillars*2:
        if i % 2 == 0:
            side = 'Left'
            pillarn = int(i/2 + 1)
        else:
            side = 'Right'
            pillarn = int((i-1)/2+1)
        df[f'Pillar {pillarn} {side} Rq (nm)'] = pillar_rq_values[i]
        i += 1
    while i < npillars*2:
        if i % 2 == 0:
            pillarn = int(i/2 + 1)
            df[f'Pillar {pillarn} Width'] = pillarwidths[int(i/2)]
        i += 1
    i = 0
    while i < npillars*2:
        if i % 2 == 0:
            pillarn = int(i/2 + 1)
            df[f'Pillar {pillarn} Duty Cycle'] = dutycycle[int(i/2)]
        i += 1 
    i = 0
    while i < npillars*2:
        if i % 2 == 0:
            pillarn = int(i/2 + 1)
            df[f'Pillar {pillarn} Width'] = pillarwidths[int(i/2)]
        i += 1

output_excel_folder = os.path.join(directory_in_str, foldername)
os.makedirs(output_excel_folder, exist_ok=True)
    
excel_output_path = os.path.join(output_excel_folder, f"{foldername} Pillar Characterization 152C60s.xlsx")

try:
    # Use 'with' statement for pd.ExcelWriter to ensure proper saving and closing
    with pd.ExcelWriter(excel_output_path, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name="Pillars", index=False)
        print(f"\nDataFrame successfully written to '{excel_output_path}'")
except Exception as e:
    print(f"\nError writing main DataFrame to Excel: {e}")