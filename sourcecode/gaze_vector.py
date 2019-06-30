import numpy as np
import cv2 
import matplotlib.pyplot as plt

#Fonction permettant de mettre l'ellipse obtenue grace au fitellipse dans le repère de l'image pour ensuite procéder au traitement

def transf_ellipse(rect):

        h = rect[0]
        H = rect[1]
        psi  = rect[2]
        cx = rect[3]
        cy = rect [4]
        
        print("psiiiiiii :")
        print(psi)

        c = np.sqrt((H/2)**2-(h/2)**2)
        b = (h/2)**2/np.sqrt((H/2)**2-(h/2)**2)

        a1 = (4/H**2)*np.cos(psi)**2 + (4/h**2)*np.sin(psi)**2
        h1 = (4/H**2)*np.cos(psi)*np.sin(psi)-(4/h**2)*np.cos(psi)*np.sin(psi)
        b1 = (4/H**2)*np.sin(psi)**2 + (4/h**2)*np.cos(psi)**2
        g1 = (4/H**2)*(cx*np.cos(psi)-(c+b)*np.sin(psi)*np.cos(psi))+((4/h**2)*(-cy*np.sin(psi)+(b+c)*np.cos(psi)*np.sin(psi)))
        f1 = (4/H**2)*(cx*np.sin(psi)-(c+b)*np.sin(psi)*np.sin(psi))+((4/h**2)*(cy*np.cos(psi)-(c+b)*np.cos(psi)*np.cos(psi)))
        d1 = (4/H**2)*(cx**2-2*cx*(b+c)*np.sin(psi)+(c+b)**2*np.sin(psi)*np.sin(psi))+(4/h**2)*(cy**2-2*cy*(b+c)*np.cos(psi)+(c+b)**2*np.cos(psi)*np.cos(psi))-1

        E = [a1,h1,b1,g1,f1,d1]

        return(E)

#Fonction de traitement de l'ellipse. Retourne le plan coupant l'oeil pour former le cercle de la pupille 

def transf_matrix(E):


#Ellipse coefficient 

	a1,h1,b1,g1,f1,d1 = E[0],E[1],E[2],E[3],E[4],E[5] 



#Vertex of the cone that hase the ellipse as a base

#Les valeurs sont calculées en fonction de la taille de l'image et d'une valeur arbitraire de la focale de la caméra utilisée

	alpha = 141
	beta = 77
	gam = 15

	vtx = (alpha, beta, gam)



#General cone equation coefficient 

	a,b = gam*gam*a1,gam*gam*b1
	c = a1*alpha**2+2*h1*alpha*beta+b1*beta**2+2*g1*alpha+2*f1*beta+d1
	d = (gam**2)*d1
	f = -gam*(b1*beta+h1*alpha+f1)
	g = -gam*(h1*beta+a1*alpha+g1)
	h = gam**2*h1
	u = gam**2*g1
	v = gam**2*f1
	w = -gam*(f1*beta+g1*alpha+d1)



#Coefficient of the cubic equation 

	A = -(a+b+c)
	B = (b*c+c*a+a*b-f**2-g**2-h**2)
	C =  -(a*b*c+2*f*g*h-a*f**2-b*g**2-c*h**2)



#Roots of the cubic equation 

	coef = np.array([1,A,B,C])

	root = np.roots(coef)
	
	print(len(root))
	
	if len(root) != 2 : 

		lbda1,lbda2,lbda3 = root[0],root[1],root[2] 
		lbda = [lbda1,lbda2,lbda3]
	else : 

		lbda1, lbda2, lbda3 = root[0],root[0],root[1]

#Coefficient of the plane cutting through the pupile 



	#Case 1 : l = 0 ; lambda 1 < lambda 2 (attention m peut etre en + ou -)


	if lbda1 < lbda2 :

		n = np.sqrt((lbda1 - lbda3)/(lbda2 - lbda3))
		m = np.sqrt((lbda2 - lbda1)/(lbda2 - lbda3))
		l = 0

	#Case 2 : lambda 1 > lambda 2 (att l peut être en + ou -)

	elif lbda1 > lbda2 : 

		n = np.sqrt((lbda2 - lbda3)/(lbda1 - lbda3))
		m = 0
		l = np.sqrt((lbda1 - lbda2)/(lbda1 - lbda3))

	#Case 3 : lambda1 = lambda2

	elif lbda1 == lbda2 :
	
		n = 1
		m = 0
		l = 0
        norm_vect = [l,m,n]	
#Rotational transformation elements 
	
	lbda = [lbda1,lbda2,lbda3]
	
	t1 = [0,0,0]
	t2 = [0,0,0]
	t3 = [0,0,0]

	L = [0,0,0]
	M = [0,0,0]
	N = [0,0,0]

	for i in range (0,3):

		t1[i] = (b - lbda[i])*g - f*h
		t2[i] = (a - lbda[i])*f - g*h
		t3[i] = - (a - lbda[i])*(t1[i]/t2[i])/g - h/g
	
		M[i] = 1/np.sqrt(1 + (t1[i]/t2[i])**2 + t3[i]**2)
		L[i] = (t1[i]/t2[i])*M[i]	
		N[i] = t3[i]*M[i]

#Rotational transformation matrix 

	T = np.array([[L[0] , L[1] , L[2] , -(u*L[0] + v*M[0] + w*N[0])/lbda[0]],
		      [M[0] , M[1] , M[2] , -(u*L[1] + v*M[1] + w*N[1])/lbda[1]],
		      [N[0] , N[1] , N[2] , -(u*L[2] + v*M[2] + w*N[2])/lbda[2]],
		      [0, 0, 0, 1]])
	return(T,norm_vect)
