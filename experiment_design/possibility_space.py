from decimal import Decimal
from math import comb

n = 11 * 11

exc = round(121 * 0.2)
inh = round(121 * 0.4)

print(f"Possible connections: {n}")
print(f"Excitatory connections: {exc}")
print(f"Inhibitory connections: {inh}")

connections = exc + inh


# modulation
pot = round(connections * 0.1)
dep = round(connections * 0.05)

print(f"\nPotentiations drawn: {pot} from {connections} excitatory connections")
print(f"Depressions drawn: {dep} from {connections} inhibitory connections")    

pot_space = comb(connections, pot)
dep_space = comb(connections, dep)

print(f"\nPotentiation possibility space C({connections},{pot}): {pot_space:.3e}")
print(f"Depression possibility space C({connections},{dep}): {dep_space:.3e}")

total_space = connections * connections * pot_space * dep_space
print(f"\nTotal possibility space: {total_space:.3e}")

# How many samples for 1% coverage?
coverage = total_space * 0.01
print(f"Samples needed for 1% coverage: {coverage:.3e}")

# How much coverage with 1000 samples?
samples = 1000 
coverage_1000 = Decimal(samples) / Decimal(total_space)
print(f"Coverage with {samples} samples: {coverage_1000:.3e}")