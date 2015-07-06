#!/usr/bin/env python3
from math import pi,atan2,sqrt,cos,sin
from sys import argv
from numpy.polynomial.polynomial import polyval
from numpy import genfromtxt,savetxt,shape,column_stack
import os

################################################################################
#                                                                              #
# Convert an aei file with kartesian state vectors x,y,z,vx,vy,vz to an aei    #
# file with orbital elements. The conversion includes the zontal gravitational #
# parameters J2, J4, J6. Implementation follows Renner, S. & Sicardy, B. 2006, #
# Celestial Mechanics and Dynamical Astronomy, 94, 237                         #
#                                                                              #
# Usage:                                                                       #
#  ./k2o.py body1.aei [body2.aei...]                                           #
# Output:                                                                      #
#  ./body1_oe.aei [body2_oe.aei...]                                            #
#                                                                              #
################################################################################

### Constants ##################################################################
J2 = 16298.0e-6
J4 = -915.0e-6
J6 = 103.0e-6
GMp = 8.15276780535E+01
Rp = 0.4 #Radius of central body
ε = 10**(-10) #convergence criterion

### Helper Functions ###########################################################
def atan2pi(y,x):
	return zweiPi(atan2(y,x))

def zweiPi(a):
	return a%(2.0*pi)

def calc_r(x,y):
	return sqrt(x**2+y**2)

def calc_L(x,y):
	return atan2pi(y,x)

def calc_rdot(vx,vy,L):
	return vx*cos(L)+vy*sin(L)

def calc_Ldot(vx,vy,L,r):
	return (-vx*sin(L)+vy*cos(L))/r

def calc_a(r,rc,ldot,lcdot,n):
	return (r-rc)/(1.0-(ldot-lcdot-n)/(2.0*n))

def calc_e(rdot,rcdot,ldot,lcdot,n,a,κ):
	return sqrt(((ldot-lcdot-n)/(2.0*n))**2 + ((rdot-rcdot)/(a*κ))**2)

def calc_I(z,zc,zdot,zcdot,a,ν):
	return sqrt(((z-zc)/a)**2 + ((zdot-zcdot)/(a*ν))**2)

def calc_λ(l,lc,rdot,rcdot,n,κ,a):
	return zweiPi(l-lc-2.0*n/κ*(rdot-rcdot)/(a*κ))

def calc_ω(λ,r,rc,rdot,rcdot,a,κ):
	return zweiPi(λ-atan2(rdot-rcdot,a*κ*(1.0-(r-rc)/a)))

def calc_Ω(λ,ν,z,zc,zdot,zcdot):
	return zweiPi(λ-atan2(ν*(z-zc),zdot-zcdot))

def GG(a):
	global GMp
	return GMp/a**3

def calc_n(a,e,I,gg):
	global Rp,J2,J4,J6
	X = Rp/a
	return sqrt(gg)*polyval(X,[1, 0, 3./4.*J2 + 3.*J2*e**2 - 12*J2*I**2, 0, 15./16.*J4 - 9./32.*J2**2, 0, 35./32.*J6 + 45./64.*J2*J4 + 27./128.*J2**3])

def calc_κ(a,e,I,gg):
	global Rp,J2,J4,J6
	X = Rp/a
	return sqrt(gg)*polyval(X,[1, 0, -3./4.*J2 - 9.*J2*I**2, 0, 45./16.*J4 - 9./32.*J2**2, 0, -175./32.*J6 + 135./64.*J2*J4 - 27./128.*J2**3])

def calc_ν(a,e,I,gg):
	global Rp,J2,J4,J6
	X = Rp/a
	return sqrt(gg)*polyval(X,[1, 0, 9./4.*J2 + 6.*J2*e**2 - 51./4.*J2*I**2, 0, -75./16.*J4 - 81./32.*J2**2, 0, 245./32.*J6 + 675./64.*J2*J4 + 729./128.*J2**3])

def calc_η_2(a,e,I,gg):
	global Rp,J2,J4,J6
	X = Rp/a
	return gg*polyval(X,[1, 0, 2*J2, 0, 75./8.*J4, 0, -175./8.*J6])

def calc_χ_2(a,e,I,gg):
	global Rp,J2,J4,J6
	X = Rp/a
	return gg*polyval(X,[1, 0, 15./2.*J2, 0, 175./8.*J4, 0, 735./16.*J6])

def calc_α1(ν,κ):
	return 1./3.*(2.*ν+κ)

def calc_α2(ν,κ):
	return 2.*ν-κ

def calc_α_2(α1,α2):
	return α1*α2

def calc_rc(a,e,I,ω,Ω,λ,η_2,κ,χ_2,α_2):
	return a*e**2*(3./2.*η_2/κ**2-1.-η_2/(2.*κ**2)*cos(2.*(λ-ω)))+a*I**2*(3./4.*χ_2/κ**2-1.+χ_2/(4.*α_2)*cos(2.*(λ-Ω)))

def calc_lc(e,I,ω,Ω,λ,η_2,κ,χ_2,α_2,n,ν):
	return e**2*(3./4.+η_2/(2.*κ**2))*n/κ*sin(2.*(λ-ω))-I**2*χ_2/(4.*α_2)*n/ν*sin(2.*(λ-Ω))

def calc_zc(a,e,I,ω,Ω,λ,κ,χ_2,α1,α2):
	return a*I*e*(χ_2/(2.*κ*α1)*sin(2.*λ-ω-Ω)-3./2.*χ_2/(κ*α2)*sin(ω-Ω))

def calc_rcdot(a,e,I,ω,Ω,λ,η_2,κ,χ_2,α_2,ν):
	return a*e**2*η_2/κ*sin(2.*(λ-ω))-a*I**2*χ_2/(2*α_2)*ν*sin(2.*(λ-Ω))

def calc_lcdot(e,I,ω,Ω,λ,η_2,κ,χ_2,α_2,n):
	return e**2*n*(7./2.-3.*η_2/κ**2-κ**2/(2.*n**2)+(3./2.+η_2/κ**2)*cos(2.*(λ-ω)))+I**2*n*(2.-κ**2/(2.*n**2)-3./2.*χ_2/κ**2-χ_2/(2.*α_2)*cos(2.*(λ-Ω)))

def calc_zcdot(a,e,I,ω,Ω,λ,κ,χ_2,ν,α1,α2):
	return a*I*e*(χ_2*(κ+ν)/(2.*κ*α1)*cos(2.*λ-ω-Ω)+3./2.*χ_2*(κ-ν)/(κ*α2)*cos(ω-Ω))

def print_orbital(a,e,I,ω,Ω,λ):
	print("a = {a: 1.15E}, e = {e: 1.15E}, I = {I: 1.15E}, ω = {om: 1.15E}, Ω = {Om: 1.5E}, λ = {la: 1.15E}".format(a=a, e=e, I=I, om=ω, Om=Ω, la=λ))

def print_kartesian(x,y,z,vx,vy,vz):
	print("x="+str(x)+", y="+str(y)+", z="+str(z)+", vx="+str(vx)+", vy="+str(vy)+", vz="+str(vz))

### Main Conversion Method #####################################################
def kartesian_to_orbital(x,y,z,vx,vy,vz):
	global GMp,Rp,J2,J4,J6,ε

	r=calc_r(x,y)
	l=calc_L(x,y)
	rdot=calc_rdot(vx,vy,l)
	ldot=calc_Ldot(vx,vy,l,r)
	zdot=vz

	#initial values
	a=r
	e=0
	I=0
	ω=0
	Ω=0
	λ=0
	rc=0
	lc=0
	zc=0
	rcdot=0
	lcdot=0
	zcdot=0

	#0th round
	lasta = a
	gg = GG(a)
	n = calc_n(a,e,I,gg)
	κ = calc_κ(a,e,I,gg)
	ν = calc_ν(a,e,I,gg)
	η_2 = calc_η_2(a,e,I,gg)
	χ_2 = calc_χ_2(a,e,I,gg)
	α1 = calc_α1(ν,κ)
	α2 = calc_α2(ν,κ)
	α_2 = calc_α_2(α1,α2)

	a = calc_a(r,rc,ldot,lcdot,n)
	e = calc_e(rdot,rcdot,ldot,lcdot,n,a,κ)
	I = calc_I(z,zc,zdot,zcdot,a,ν)
	λ = calc_λ(l,lc,rdot,rcdot,n,κ,a)
	ω = calc_ω(λ,r,rc,rdot,rcdot,a,κ)
	Ω = calc_Ω(λ,ν,z,zc,zdot,zcdot)

	#print_orbital(a,e,I,ω,Ω,λ)

	#start loop
	max_loops=1e4
	while abs(a-lasta) > ε:
		lasta = a

		try:

			#helper parameters
			rc = calc_rc(a,e,I,ω,Ω,λ,η_2,κ,χ_2,α_2)
			lc = calc_lc(e,I,ω,Ω,λ,η_2,κ,χ_2,α_2,n,ν)
			zc = calc_zc(a,e,I,ω,Ω,λ,κ,χ_2,α1,α2)
			rcdot = calc_rcdot(a,e,I,ω,Ω,λ,η_2,κ,χ_2,α_2,ν)
			lcdot = calc_lcdot(e,I,ω,Ω,λ,η_2,κ,χ_2,α_2,n)
			zcdot = calc_zcdot(a,e,I,ω,Ω,λ,κ,χ_2,ν,α1,α2)
	
			#frequencies
			gg = GG(a)
			n = calc_n(a,e,I,gg)
			κ = calc_κ(a,e,I,gg)
			ν = calc_ν(a,e,I,gg)
			η_2 = calc_η_2(a,e,I,gg)
			χ_2 = calc_χ_2(a,e,I,gg)
			α1 = calc_α1(ν,κ)
			α2 = calc_α2(ν,κ)
			α_2 = calc_α_2(α1,α2)
	
			#orbital elements
			a = calc_a(r,rc,ldot,lcdot,n)
			e = calc_e(rdot,rcdot,ldot,lcdot,n,a,κ)
			I = calc_I(z,zc,zdot,zcdot,a,ν)
			λ = calc_λ(l,lc,rdot,rcdot,n,κ,a)
			ω = calc_ω(λ,r,rc,rdot,rcdot,a,κ)
			Ω = calc_Ω(λ,ν,z,zc,zdot,zcdot)

			#print_orbital(a,e,I,ω,Ω,λ)
		except ValueError:
			print("    conversion failed (domain error)")
			return (-1,-1,-1,-1,-1,-1)
		max_loops-=1
		if(max_loops==0):
			print("    conversion failed (no convergence in 1e4 iterations)")
			return (-1,-1,-1,-1,-1,-1)

	return (a,e,I,ω,Ω,λ)

### Main Program ###############################################################
if len(argv) < 2:
	print("Give at least one file to convert!")

else:
	g=lambda x: (x['Time_years'] if 'Time_years' in x.dtype.names else x['Time_days']/365.25,)+kartesian_to_orbital(x['x'],x['y'],x['z'],x['vx'],x['vy'],x['vz'])+(sqrt(x['x']**2+x['y']**2+x['z']**2),)
	#for every file in the command line argument
	for datei in argv[1:]:
		saveto=os.path.splitext(datei)[0]+"_oe.aei"
		name=os.path.splitext(os.path.basename(datei))[0]
		data=genfromtxt(datei,skip_header=3,names=True)
		print("converting "+name+" now...")
		if shape(data) != ():
			converted=list(map(g,data))
		else:
			converted=column_stack(g(data))
		savetxt(saveto,converted,header=name+"\na: semi-major axis, e: eccentricity, I: inclination, om: longitude of pericentre, OM: longitude of ascending node, lam: mean longitude, r: distance from saturn\nTime_years\ta\te\tI\tom\tOM\tlam\tr")
		print(name+" done.")

