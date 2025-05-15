pi = 3.14159265358979323846264338327950288419716939937510 # Quelle Wikipedia (50 Stellen)
h = 6.62607554E-34  # J*s (Quelle molpro2020)
m_e = 9.1093897E-31  # kg
e = 1.60217733E-19  # C
c = 2.99792458E+10  # m/s (Quelle molpro2020)
h_quer = h / (2 * pi)
hartree = 4.3597482E-18  # 4.3597447222071E-18 #J (Quelle molpro2020)
avogadro = 6.022136736E23  # mol-1 = R/k_B (Quelle molpro2020)
epsilon_0 = 8.854187817E-12  # As/(Vm) (Quelle molpro2020)
jpcal = 4.184E+00  # joule per calory (Quelle molpro2020); vorher 4.1868

bohr_radius = 5.2917721092e-11  # m; Wert laut /opt/molpro-mpp-2024/lib/include/codata/molpro2020
a_to_bohr_molpro = 1E-10 / bohr_radius  # 1.889 726 124 565 061 8


from scipy.constants import c, epsilon_0, h, Avogadro, G, pi
print(f"Lichtgeschwindigkeit: {c} m/s")
print(f"epsilon_0: {epsilon_0} F/m")
print(f"Planck-Konstante: {h} J*s")
print(f"Avogadro-Zahl: {Avogadro} 1/mol")
print(f"Gravitationskonstante: {G} m^3/kg/s^2")
print(f"pi: {pi}")

print(1/(4*pi*epsilon_0))