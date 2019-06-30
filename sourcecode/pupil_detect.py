import numpy as np 
import cv2
import matplotlib.pyplot as plt 
import gaze_vector as gz
import complete as cp 

image = cv2.imread('Dark_pupil.jpg',0)
ret,thresh = cv2.threshold(image,120,255,cv2.THRESH_BINARY)
im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
cv2.imshow('tresh',thresh)
cv2.waitKey(0)
#if len(contours) != 0:	
#	for cont in contours:
#		
#		print(len(contours))		
#		elps = cv2.fitEllipse(cont)
#		print(elps)
#		cv2.ellipse(image,elps,(0,255,0),1)
#print ("contours :")
print (contours)
print (len(contours))
#contours2 =[]
#i=-1
#for i in range(0,len(contours)) :
#
#    if len(contours[i]) > 6 :
#        print("coucou")
#        print(contours[i])
#        contours2 +=[contours[i]]
#
#
#print(len(contours2))
elps1 = cv2.fitEllipse(contours[1])
print(elps1)



#################################################################################################

#Dessin de la pupille? 

cv2.circle(image,(int(elps1[0][0]),int(elps1[0][1])),0,(255,0,0),1)

#################################################################################################




#################################################################################################

#Coef obtenus par le fitellipse

cx = elps1[0][0]
cy = elps1[0][1]
l = elps1[1][0]
h = elps1[1][1]
psi =elps1[2]

#################################################################################################



#################################################################################################

#dessin du rectangle 

	#point extreme bas droit

pebx = int(cx - h/2*np.sin(psi) + l/2*np.cos(psi))
peby = int(cy + h/2*np.cos(psi) + l/2*np.sin(psi) )

	#point extreme haut gauche

pehx = int(cx + h/2*np.sin(psi) - l/2*np.cos(psi))
pehy = int(cy - h/2*np.cos(psi) - l/2*np.sin(psi) )

#cv2.rectangle(image,(pebx,peby),(pehx,pehy),(255,0,0),2)

cv2.ellipse(image,elps1,(0,255,0),1)

#cv2.drawContours(image, contours, -1, (0,255,0), 1)
#################################################################################################


#################################################################################################

#Calcul du plan coupant l'oeil en un cercle constitué de la pupille

rect = [l,h,psi,cx,cy]
Ellipse = []
Transformee = []

Ellipse  = gz.transf_ellipse(rect)

print("Ellipse =")
print(Ellipse)

print("Trans")
[Trans,n] = gz.transf_matrix (Ellipse)

print(Trans)

coord_centre = [[cx],
		[cy],
		[1],
                [1]]

A = [cx,cy] #centre de la pupille

rot = [[Trans[0][0],Trans[0][1],Trans[0][2]],
       [Trans[1][0],Trans[1][1],Trans[1][2]],
       [Trans[2][0],Trans[2][1],Trans[2][2]]]

rot_rnd = np.array([[0.536,0.536,0.411],
                    [0.673,0.673,0.477],
                    [-0.507,-0.507,-0.776]])

rot_rnd_inv = np.linalg.pinv(rot_rnd)

print("inverse de 3x3")

print(rot_rnd_inv)

print(rot_rnd*rot_rnd_inv)


rot_inv = np.linalg.pinv(rot)

trans_inv = np.array([[rot_inv[0][0],rot_inv[0][1],rot_inv[0][2],(-Trans[0][3])],
            [rot_inv[1][0],rot_inv[1][1],rot_inv[1][2],(-Trans[1][3])],
            [rot_inv[2][0],rot_inv[2][1],rot_inv[2][2],(-Trans[2][3])],
            [0,0,0,1]])

print("inverse =")
print(trans_inv)

#B = rot_inv @ coord_centre
B = Trans @ coord_centre
print ("A =")
print(A)

print("B =")
print(B)

Bx = int(B[0][0])
By = int(B[1][0])

print("B=")
print(Bx)

#On dessine la projection du vecteur regard dans le plan de l'image

cv2.line(image,(int(cx),int(cy)),(cx+n[0],cy+n[1]),(255,0,0),1)

#################################################################################################
cv2.namedWindow('img',cv2.WINDOW_NORMAL)
cv2.resizeWindow('img', 600,600)
cv2.imshow('img',image)
cv2.waitKey(0)
