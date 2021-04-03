from panorama import Stitcher
import cv2
import numpy as np
import os

sequences = ['legumes', 'magasin', 'neige', 'parc', 'studio', 'visages']

stitcher  = Stitcher()
try:
    os.mkdir('Resultats Stitch')
except OSError as error:
    print(error)

# Chargement des images des une liste 2D
for i, sequence in enumerate(sequences) :
    index = 0
    try :
        os.mkdir('Resultats Stitch/' + sequence)
    except OSError as error :
        print(error)

    while True:
        path1 = 'SYS809_projet2021_sequences/' + sequence + 'A-' + str(index).zfill(2) + '.jpg'
        path2 = 'SYS809_projet2021_sequences/' + sequence + 'A-' + str(index+1).zfill(2) + '.jpg'
        image1 = cv2.imread(path1)
        image2 = cv2.imread(path2)

        if image2 is not None:
            (result, vis) = stitcher.stitch([image1, image2], ratio=0.8, showMatches=True)
            (hA, wA) = vis.shape[:2]
            (hB, wB) = result.shape[:2]
            total = np.zeros((hA + hB, max(wA, wB), 3), dtype="uint8")
            total[0:hA, 0:wA] = vis
            total[hA:, 0:wA] = result

            cv2.imwrite('Resultats Stitch/' + sequence + '/' + str(index).zfill(2) + '-' + str(index+1).zfill(2) + '.jpg', total)

            (result, vis) = stitcher.stitch([image2, image1], ratio=0.8, showMatches=True)
            (hA, wA) = vis.shape[:2]
            (hB, wB) = result.shape[:2]
            total = np.zeros((hA + hB, max(wA, wB), 3), dtype="uint8")
            total[0:hA, 0:wA] = vis
            total[hA:, 0:wA] = result

            cv2.imwrite('Resultats Stitch/' + sequence + '/' + str(index+1).zfill(2) + '-' + str(index).zfill(2) + '.jpg', total)

            index += 1
        else:
            break