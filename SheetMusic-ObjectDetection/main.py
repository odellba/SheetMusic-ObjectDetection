import sys
import subprocess
import cv2
import time
import numpy as np
from best_fit import fit
from rectangle import Rectangle
from note import Note
from random import randint
from compare import compare
#from midiutil.MidiFile3 import MIDIFile
from midiutil import MIDIFile
#images for object detection
staff_files = [
    "resources/template/staff2.png", 
    "resources/template/staff.png"]
quarter_files = [
    "resources/template/quarter.png", 
    "resources/template/solid-note.png"]
sharp_files = [
    "resources/template/sharp.png"]
flat_files = [
    "resources/template/flat-line.png", 
    "resources/template/flat-space.png" ]
half_files = [
    "resources/template/half-space.png", 
    "resources/template/half-note-line.png",
    "resources/template/half-line.png", 
    "resources/template/half-note-space.png"]
whole_files = [
    "resources/template/whole-space.png", 
    "resources/template/whole-note-line.png",
    "resources/template/whole-line.png", 
    "resources/template/whole-note-space.png"]

staff_imgs = [cv2.imread(staff_file, 0) for staff_file in staff_files]
quarter_imgs = [cv2.imread(quarter_file, 0) for quarter_file in quarter_files]
sharp_imgs = [cv2.imread(sharp_files, 0) for sharp_files in sharp_files]
flat_imgs = [cv2.imread(flat_file, 0) for flat_file in flat_files]
half_imgs = [cv2.imread(half_file, 0) for half_file in half_files]
whole_imgs = [cv2.imread(whole_file, 0) for whole_file in whole_files]

def locate_images(img, templates, start, stop, threshold):
    locations, scale = fit(img, templates, start, stop, threshold)
    img_locations = []
    for i in range(len(templates)):
        w, h = templates[i].shape[::-1]
        w *= scale
        h *= scale
        img_locations.append([Rectangle(pt[0], pt[1], w, h) for pt in zip(*locations[i][::-1])])
    return img_locations

def merge_recs(recs, threshold):
    filtered_recs = []
    while len(recs) > 0:
        r = recs.pop(0)
        recs.sort(key=lambda rec: rec.distance(r))
        merged = True
        while(merged):
            merged = False
            i = 0
            for _ in range(len(recs)):
                if r.overlap(recs[i]) > threshold or recs[i].overlap(r) > threshold:
                    r = r.merge(recs.pop(i))
                    merged = True
                elif recs[i].distance(r) > r.w/2 + recs[i].w/2:
                    break
                else:
                    i += 1
        filtered_recs.append(r)
    return filtered_recs
#opens file
def open_file(path):
    cmd = {'linux':'eog', 'win32':'explorer', 'darwin':'open'}[sys.platform]
    subprocess.run([cmd, path])

if __name__ == "__main__":
    img_file = sys.argv[1:][0]
    img = cv2.imread(img_file, 0)
    img_gray = img#cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img_gray,cv2.COLOR_GRAY2RGB)
    ret,img_gray = cv2.threshold(img_gray,127,255,cv2.THRESH_BINARY)
    img_width, img_height = img_gray.shape[::-1]
    #parameters for sizing for object detection
    staff_lower, staff_upper, staff_thresh = 50, 150, 0.73
    sharp_lower, sharp_upper, sharp_thresh = 50, 150, 0.70
    flat_lower, flat_upper, flat_thresh = 50, 150, 0.77
    quarter_lower, quarter_upper, quarter_thresh = 50, 150, 0.73
    half_lower, half_upper, half_thresh = 50, 150, 0.73
    whole_lower, whole_upper, whole_thresh = 50, 150, 0.73
    #Commented out section could be used to iterate through all parameters to find best(not used currently/will take awhile)
    """
    #lower loop
    for a in range(35, 100):
        #upper loop
        for b in range(100, 200):
            #staff threshhold loop
            for c in np.arange(0.5, 0.9, 0.01):
                #sharp thresh loop
                for d in np.arange(0.5, 0.9, 0.01):
                    #flat thesh loop
                    for e in np.arange(0.5, 0.9, 0.01):
                        #quater loop
                        for f in np.arange(0.5, 0.9, 0.01):
                            #half thresh loop
                            for g in np.arange(0.5, 0.9, 0.01):
                                #whole thresh loop
                                for h in np.arange(0.5, 0.9, 0.01):"""
    print("Matching staff image...")
    #locates staffs
    staff_recs = locate_images(img_gray, staff_imgs, staff_lower, staff_upper, staff_thresh)

    print("Filtering weak staff matches...")
    staff_recs = [j for i in staff_recs for j in i]
    heights = [r.y for r in staff_recs] + [0]
    histo = [heights.count(i) for i in range(0, max(heights) + 1)]
    avg = np.mean(list(set(histo)))
    staff_recs = [r for r in staff_recs if histo[r.y] > avg]

    print("Merging staff image results...")
    #Draws on original image where staffs were recognized
    staff_recs = merge_recs(staff_recs, 0.01)
    staff_recs_img = img.copy()
    for r in staff_recs:
        r.draw(staff_recs_img, (0, 0, 255), 2)
    cv2.imwrite('staff_recs_img.png', staff_recs_img)
    open_file('staff_recs_img.png')

    print("Discovering staff locations...")
    #Draws boxes on original image where staffs were recognized for each line of music
    staff_boxes = merge_recs([Rectangle(0, r.y, img_width, r.h) for r in staff_recs], 0.01)
    staff_boxes_img = img.copy()
    for r in staff_boxes:
        r.draw(staff_boxes_img, (0, 0, 255), 2)
    cv2.imwrite('staff_boxes_img.png', staff_boxes_img)
    open_file('staff_boxes_img.png')
    
    print("Matching sharp image...")
    #locates sharps
    sharp_recs = locate_images(img_gray, sharp_imgs, sharp_lower, sharp_upper, sharp_thresh)

    print("Merging sharp image results...")
    #Draws boxes on original image where sharps were recognized
    sharp_recs = merge_recs([j for i in sharp_recs for j in i], 0.5)
    sharp_recs_img = img.copy()
    for r in sharp_recs:
        r.draw(sharp_recs_img, (0, 0, 255), 2)
    cv2.imwrite('sharp_recs_img.png', sharp_recs_img)
    open_file('sharp_recs_img.png')

    print("Matching flat image...")
    #locates flats
    flat_recs = locate_images(img_gray, flat_imgs, flat_lower, flat_upper, flat_thresh)

    print("Merging flat image results...")
    #Draws boxes on original image where flats were recognized
    flat_recs = merge_recs([j for i in flat_recs for j in i], 0.5)
    flat_recs_img = img.copy()
    for r in flat_recs:
        r.draw(flat_recs_img, (0, 0, 255), 2)
    cv2.imwrite('flat_recs_img.png', flat_recs_img)
    open_file('flat_recs_img.png')

    print("Matching quarter image...")
    #locates quarter notes
    quarter_recs = locate_images(img_gray, quarter_imgs, quarter_lower, quarter_upper, quarter_thresh)

    print("Merging quarter image results...")
    #Draws boxes on original image where quarter notes were recognized
    quarter_recs = merge_recs([j for i in quarter_recs for j in i], 0.5)
    quarter_recs_img = img.copy()
    for r in quarter_recs:
        r.draw(quarter_recs_img, (0, 0, 255), 2)
    cv2.imwrite('quarter_recs_img.png', quarter_recs_img)
    open_file('quarter_recs_img.png')

    print("Matching half image...")
    #locates half notes
    half_recs = locate_images(img_gray, half_imgs, half_lower, half_upper, half_thresh)

    print("Merging half image results...")
    #Draws boxes on original image where half notes were recognized
    half_recs = merge_recs([j for i in half_recs for j in i], 0.5)
    half_recs_img = img.copy()
    for r in half_recs:
        r.draw(half_recs_img, (0, 0, 255), 2)
    cv2.imwrite('half_recs_img.png', half_recs_img)
    open_file('half_recs_img.png')

    print("Matching whole image...")
    #locates whole notes
    whole_recs = locate_images(img_gray, whole_imgs, whole_lower, whole_upper, whole_thresh)

    print("Merging whole image results...")
    #Draws boxes on original image where whole notes were recognized
    whole_recs = merge_recs([j for i in whole_recs for j in i], 0.5)
    whole_recs_img = img.copy()
    for r in whole_recs:
        r.draw(whole_recs_img, (0, 0, 255), 2)
    cv2.imwrite('whole_recs_img.png', whole_recs_img)
    open_file('whole_recs_img.png')

    
    best_score=0
    best_step=0
    #beginning of loop to determine best note_step
    for i in np.arange(0.05, 0.2, 0.0001):
        note_groups = []
        note_step=i
        for box in staff_boxes:
            #defines all recognized notes
            staff_sharps = [Note(note_step,r,"sharp", box) 
                for r in sharp_recs if abs(r.middle[1] - box.middle[1]) < box.h*5.0/8.0]
            staff_flats = [Note(note_step,r, "flat", box) 
                for r in flat_recs if abs(r.middle[1] - box.middle[1]) < box.h*5.0/8.0]
            quarter_notes = [Note(note_step,r, "4,8", box, staff_sharps, staff_flats) 
                for r in quarter_recs if abs(r.middle[1] - box.middle[1]) < box.h*5.0/8.0]
            half_notes = [Note(note_step,r, "2", box, staff_sharps, staff_flats) 
                for r in half_recs if abs(r.middle[1] - box.middle[1]) < box.h*5.0/8.0]
            whole_notes = [Note(note_step,r, "1", box, staff_sharps, staff_flats) 
                for r in whole_recs if abs(r.middle[1] - box.middle[1]) < box.h*5.0/8.0]
            staff_notes = quarter_notes + half_notes + whole_notes
            staff_notes.sort(key=lambda n: n.rec.x)
            staffs = [r for r in staff_recs if r.overlap(box) > 0]
            staffs.sort(key=lambda r: r.x)
            note_color = (randint(0, 255), randint(0, 255), randint(0, 255))
            note_group = []
            i = 0; j = 0;
            #appends note_def to note_group, colors the box around the note
            while(i < len(staff_notes)):
                try:
                    if (staff_notes[i].rec.x > staffs[j].x and j < len(staffs)):
                        r = staffs[j]
                        j += 1;
                        if len(note_group) > 0:
                            note_groups.append(note_group)
                            note_group = []
                        note_color = (randint(0, 255), randint(0, 255), randint(0, 255))
                    else:
                        note_group.append(staff_notes[i])
                        staff_notes[i].rec.draw(img, note_color, 2)
                        i += 1
                except IndexError:
                    print("Exception caught for list index out of range")
                    note_group.append(staff_notes[i])
                    staff_notes[i].rec.draw(img, note_color, 2)
                    i += 1
            note_groups.append(note_group)
        print(note_step)
        for note_group in note_groups:
            print([ note.note + " " + note.sym for note in note_group])
        #creates MIDI file
        midi = MIDIFile(1)
        
        track = 0   
        time = 0
        channel = 0
        volume = 100
        
        midi.addTrackName(track, time, "Track")
        midi.addTempo(track, time, 140)
        #Adds each note to MIDI
        for note_group in note_groups:
            duration = None
            for note in note_group:
                note_type = note.sym
                if note_type == "1":
                    duration = 4
                elif note_type == "2":
                    duration = 2
                elif note_type == "4,8":
                    duration = 1 if len(note_group) == 1 else 0.5
                pitch = note.pitch
                midi.addNote(track,channel,pitch,time,duration,volume)
                time += duration

        midi.addNote(track,channel,pitch,time,4,0)
        # And write it to disk.
        #outputs MIDI
        binfile = open("output.mid", 'wb')
        binfile.truncate()
        midi.writeFile(binfile)
        binfile.close()
        #Compares output to MuseScore generated MIDI
        Right_Test,Right_output=compare()
        #takes average
        score=(Right_Test+Right_output)/2
        print(score)
        #updates best score, best note_step found
        if score>best_score:
            best_score=score
            best_step=note_step
            """
            best_staff_lower=a
            best_sharp_lower=a
            best_flat_lower=a
            best_quarter_lower=a
            best_half_lower=a
            best_whole_lower=a
            best_staff_upper=b
            best_sharp_upper=b
            best_flat_upper=b
            best_quarter_upper=b
            best_half_upper=b
            best_whole_upper=b
            best_staff_thresh=c
            best_sharp_thresh=d
            best_flat_thresh=e
            best_quarter_thresh=f
            best_half_thresh=g
            best_whole_thresh=h
            """
    #end of loop
    print(best_score,best_step)
    #commented section below would be used to implement the best parameters found by iterating through all parameters
    """
    staff_lower, staff_upper, staff_thresh = best_staff_lower, best_staff_upper, best_staff_thresh
    sharp_lower, sharp_upper, sharp_thresh = best_sharp_lower, best_sharp_upper, best_sharp_thresh
    flat_lower, flat_upper, flat_thresh = best_flat_lower, best_flat_upper, best_flat_thresh
    quarter_lower, quarter_upper, quarter_thresh = best_quarter_lower, best_quarter_upper, best_quarter_thresh
    half_lower, half_upper, half_thresh = best_half_lower, best_half_upper, best_half_thresh
    whole_lower, whole_upper, whole_thresh = best_whole_lower, best_whole_upper, best_whole_thresh

    
    print("Matching staff image...")
    staff_recs = locate_images(img_gray, staff_imgs, staff_lower, staff_upper, staff_thresh)

    print("Filtering weak staff matches...")
    staff_recs = [j for i in staff_recs for j in i]
    heights = [r.y for r in staff_recs] + [0]
    histo = [heights.count(i) for i in range(0, max(heights) + 1)]
    avg = np.mean(list(set(histo)))
    staff_recs = [r for r in staff_recs if histo[r.y] > avg]

    print("Merging staff image results...")
    staff_recs = merge_recs(staff_recs, 0.01)
    staff_recs_img = img.copy()
    for r in staff_recs:
        r.draw(staff_recs_img, (0, 0, 255), 2)
    cv2.imwrite('staff_recs_img.png', staff_recs_img)
    open_file('staff_recs_img.png')

    print("Discovering staff locations...")
    staff_boxes = merge_recs([Rectangle(0, r.y, img_width, r.h) for r in staff_recs], 0.01)
    staff_boxes_img = img.copy()
    for r in staff_boxes:
        r.draw(staff_boxes_img, (0, 0, 255), 2)
    cv2.imwrite('staff_boxes_img.png', staff_boxes_img)
    open_file('staff_boxes_img.png')
    
    print("Matching sharp image...")
    sharp_recs = locate_images(img_gray, sharp_imgs, sharp_lower, sharp_upper, sharp_thresh)

    print("Merging sharp image results...")
    sharp_recs = merge_recs([j for i in sharp_recs for j in i], 0.5)
    sharp_recs_img = img.copy()
    for r in sharp_recs:
        r.draw(sharp_recs_img, (0, 0, 255), 2)
    cv2.imwrite('sharp_recs_img.png', sharp_recs_img)
    open_file('sharp_recs_img.png')

    print("Matching flat image...")
    flat_recs = locate_images(img_gray, flat_imgs, flat_lower, flat_upper, flat_thresh)

    print("Merging flat image results...")
    flat_recs = merge_recs([j for i in flat_recs for j in i], 0.5)
    flat_recs_img = img.copy()
    for r in flat_recs:
        r.draw(flat_recs_img, (0, 0, 255), 2)
    cv2.imwrite('flat_recs_img.png', flat_recs_img)
    open_file('flat_recs_img.png')

    print("Matching quarter image...")
    quarter_recs = locate_images(img_gray, quarter_imgs, quarter_lower, quarter_upper, quarter_thresh)

    print("Merging quarter image results...")
    quarter_recs = merge_recs([j for i in quarter_recs for j in i], 0.5)
    quarter_recs_img = img.copy()
    for r in quarter_recs:
        r.draw(quarter_recs_img, (0, 0, 255), 2)
    cv2.imwrite('quarter_recs_img.png', quarter_recs_img)
    open_file('quarter_recs_img.png')

    print("Matching half image...")
    half_recs = locate_images(img_gray, half_imgs, half_lower, half_upper, half_thresh)

    print("Merging half image results...")
    half_recs = merge_recs([j for i in half_recs for j in i], 0.5)
    half_recs_img = img.copy()
    for r in half_recs:
        r.draw(half_recs_img, (0, 0, 255), 2)
    cv2.imwrite('half_recs_img.png', half_recs_img)
    open_file('half_recs_img.png')

    print("Matching whole image...")
    whole_recs = locate_images(img_gray, whole_imgs, whole_lower, whole_upper, whole_thresh)

    print("Merging whole image results...")
    whole_recs = merge_recs([j for i in whole_recs for j in i], 0.5)
    whole_recs_img = img.copy()
    for r in whole_recs:
        r.draw(whole_recs_img, (0, 0, 255), 2)
    cv2.imwrite('whole_recs_img.png', whole_recs_img)
    open_file('whole_recs_img.png')"""
    note_step=best_step
    note_groups = []
    #defines all recognized notes for best_note_step found
    for box in staff_boxes:
        staff_sharps = [Note(note_step,r,"sharp", box) 
            for r in sharp_recs if abs(r.middle[1] - box.middle[1]) < box.h*5.0/8.0]
        staff_flats = [Note(note_step,r, "flat", box) 
            for r in flat_recs if abs(r.middle[1] - box.middle[1]) < box.h*5.0/8.0]
        quarter_notes = [Note(note_step,r, "4,8", box, staff_sharps, staff_flats) 
            for r in quarter_recs if abs(r.middle[1] - box.middle[1]) < box.h*5.0/8.0]
        half_notes = [Note(note_step,r, "2", box, staff_sharps, staff_flats) 
            for r in half_recs if abs(r.middle[1] - box.middle[1]) < box.h*5.0/8.0]
        whole_notes = [Note(note_step,r, "1", box, staff_sharps, staff_flats) 
            for r in whole_recs if abs(r.middle[1] - box.middle[1]) < box.h*5.0/8.0]
        staff_notes = quarter_notes + half_notes + whole_notes
        staff_notes.sort(key=lambda n: n.rec.x)
        staffs = [r for r in staff_recs if r.overlap(box) > 0]
        staffs.sort(key=lambda r: r.x)
        note_color = (randint(0, 255), randint(0, 255), randint(0, 255))
        note_group = []
        i = 0; j = 0;
        #appends notes to note_group and colors box
        while(i < len(staff_notes)):
            try:
                if (staff_notes[i].rec.x > staffs[j].x and j < len(staffs)):
                    r = staffs[j]
                    j += 1;
                    if len(note_group) > 0:
                        note_groups.append(note_group)
                        note_group = []
                    note_color = (randint(0, 255), randint(0, 255), randint(0, 255))
                else:
                    note_group.append(staff_notes[i])
                    staff_notes[i].rec.draw(img, note_color, 2)
                    i += 1
            except IndexError:
                print("Exception caught for list index out of range")
                note_group.append(staff_notes[i])
                staff_notes[i].rec.draw(img, note_color, 2)
                i += 1
        note_groups.append(note_group)
        #draws boxes on original image for all recognized
    for r in staff_boxes:
        r.draw(img, (0, 0, 255), 2)
    for r in sharp_recs:
        r.draw(img, (0, 0, 255), 2)
    flat_recs_img = img.copy()
    for r in flat_recs:
        r.draw(img, (0, 0, 255), 2)
        
    cv2.imwrite('res.png', img)
    open_file('res.png')
    for note_group in note_groups:
        print([ note.note + " " + note.sym for note in note_group])
    #MIDI file started
    midi = MIDIFile(1)
        
    track = 0   
    time = 0
    channel = 0
    volume = 100
        
    midi.addTrackName(track, time, "Track")
    midi.addTempo(track, time, 140)
    #each note added to MIDI  
    for note_group in note_groups:
        duration = None
        for note in note_group:
            note_type = note.sym
            if note_type == "1":
                duration = 4
            elif note_type == "2":
                duration = 2
            elif note_type == "4,8":
                duration = 1 if len(note_group) == 1 else 0.5
            pitch = note.pitch
            midi.addNote(track,channel,pitch,time,duration,volume)
            time += duration

    midi.addNote(track,channel,pitch,time,4,0)
        # And write it to disk.
        #outputs MIDI
    binfile = open("output.mid", 'wb')
    binfile.truncate()
    midi.writeFile(binfile)
    binfile.close()
    #Compares final to MuseScore MIDI
    score=compare()
    print(best_score,best_step)
    #plays final outputed MIDI
    open_file('output.mid')