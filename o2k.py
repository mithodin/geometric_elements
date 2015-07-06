#!/usr/bin/env python3
from math import sin,cos,sqrt,pi
from numpy.polynomial.polynomial import polyval
from sys import argv

################################################################################
#                                                                              #
# Convert a set of orbital elements to a cartesian state vector.               #
#                                                                              #
################################################################################

### Parameters #################################################################
J2 = 16298.0e-6
J4 = -915.0e-6
J6 = 103.0e-6
#Rp = 0.398349291878135 #Radius of central body in SAU
Rp = 0.4
GMp = 8.15276780535E+01
sau = 1.51464E+10 #in cm

### Helper Functions ###########################################################
def zweiPi(a):
	return a%(2.0*pi)

def calc_r(a,e,I,ω,Ω,λ,η_2,κ,χ_2,α_2):
	return a*(1-e*cos(λ-ω)+e**2*(3./2.*η_2/κ**2-1.-η_2/(2.*η_2)*cos(2.*(λ-ω)))+I**2*(3./4.*χ_2/κ**2-1.+χ_2/(4.*α_2)*cos(2.*(λ-Ω))))

def calc_l(a,e,I,ω,Ω,λ,η_2,κ,χ_2,α_2,n,ν):
	return λ+2.*e*n/κ*sin(λ-ω)+e**2*(3./4.+η_2/(2.*κ**2))*n/κ*sin(2.*(λ-ω))-I**2*χ_2/(4.*α_2)*n/ν*sin(2*(λ-Ω))

def calc_rdot(a,e,I,ω,Ω,λ,η_2,κ,χ_2,α_2,ν):
	return a*κ*(e*sin(λ-ω)+e**2*η_2/κ**2*sin(2.*(λ-ω))-I**2*χ_2/(2.*α_2)*ν/κ*sin(2.*(λ-Ω)))

def calc_ldrot(a,e,I,ω,Ω,λ,η_2,κ,χ_2,α_2,n):
	return n*(1+2.*e*cos(λ-ω)+e**2*(7./2.-3.*η_2/κ**2-κ**2/(2.*n**2)+(3./2.+η_2/κ**2)*cos(2.*(λ-ω)))+I**2*(2.-κ**2/(2.*n**2)-3./2.*χ_2/κ**2-χ_2/(2.*α_2)*cos(2.*(λ-Ω))))

def calc_vx(r,rdot,l,ldot):
	return rdot*cos(l)-r*ldot*sin(l)

def calc_vy(r,rdot,l,ldot):
	return rdot*sin(l)+r*ldot*cos(l)

def calc_vz(a,e,I,ω,Ω,λ,κ,χ_2,α1,α2,ν):
	return a*I*ν*(cos(λ-Ω)+e*χ_2*(κ+ν)/(2.*κ*α1*ν)*cos(2.*λ-ω-Ω)+e*3./2.*χ_2*(κ-ν)/(κ*α2*ν)*cos(ω-Ω))

def calc_x(r,l):
	return r*cos(l)

def calc_y(r,l):
	return r*sin(l)

def calc_z(a,e,I,ω,Ω,λ,κ,χ_2,α1,α2):
	return a*I*(sin(λ-Ω)+e*χ_2/(2.*κ*α1)*sin(2.*λ-ω-Ω)-e*3./2.*χ_2/(2.*κ*α2)*sin(ω-Ω))

def GG(a):
	global GMp
	return GMp/a**3

def calc_n(a,e,I,gg):
	global GMp,Rp,J2,J4,J6
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

def unpack(oevector):
	return(oevector[0],oevector[1],oevector[2],oevector[3],oevector[4],oevector[5])

### Main Conversion Function ###################################################
def orbital_to_kartesian(a,e,I,ω,Ω,λ):
	gg=GG(a)
	n = calc_n(a,e,I,gg)
	κ = calc_κ(a,e,I,gg)
	ν = calc_ν(a,e,I,gg)
	η_2 = calc_η_2(a,e,I,gg)
	χ_2 = calc_χ_2(a,e,I,gg)
	α1 = calc_α1(ν,κ)
	α2 = calc_α2(ν,κ)
	α_2 = calc_α_2(α1,α2)

	r = calc_r(a,e,I,ω,Ω,λ,η_2,κ,χ_2,α_2)
	l = calc_l(a,e,I,ω,Ω,λ,η_2,κ,χ_2,α_2,n,ν)
	rdot = calc_rdot(a,e,I,ω,Ω,λ,η_2,κ,χ_2,α_2,ν)
	ldot = calc_ldrot(a,e,I,ω,Ω,λ,η_2,κ,χ_2,α_2,n)
	vx = calc_vx(r,rdot,l,ldot)
	vy = calc_vy(r,rdot,l,ldot)
	vz = calc_vz(a,e,I,ω,Ω,λ,κ,χ_2,α1,α2,ν)
	x = calc_x(r,l)
	y = calc_y(r,l)
	z = calc_z(a,e,I,ω,Ω,λ,κ,χ_2,α1,α2)

	return(x,y,z,vx,vy,vz)

### Main Program (Test Case) ###################################################
km = 1.0E+5/sau
deg = pi/180.0
janus=[151460*km,0.0068,0.1640*deg,288.1778*deg,46.9389*deg,171.4419*deg]
epimetheus=[151410*km,0.0098,0.3524*deg,37.8567*deg,85.2616*deg,346.1286*deg]
#if len(argv) != 7:
#	print("Wrong number of arguments (a,e,I,ω,Ω,λ)!")
#
#else:
#	a = float(argv[1])
#	e = float(argv[2])
#	I = float(argv[3])
#	ω = float(argv[4])
#	Ω = float(argv[5])
#	λ = zweiPi(float(argv[6]))
#
a,e,I,ω,Ω,λ=unpack(janus)
x,y,z,vx,vy,vz = orbital_to_kartesian(a,e,I,ω,Ω,λ)
print("Janus: x="+str(x)+", y="+str(y)+", z="+str(z)+", vx="+str(vx)+", vy="+str(vy)+", vz="+str(vz))

a,e,I,ω,Ω,λ=unpack(epimetheus)
x,y,z,vx,vy,vz = orbital_to_kartesian(a,e,I,ω,Ω,λ)
print("Epimetheus: x="+str(x)+", y="+str(y)+", z="+str(z)+", vx="+str(vx)+", vy="+str(vy)+", vz="+str(vz))

