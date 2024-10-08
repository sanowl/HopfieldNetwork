import numpy as np
from typing import List, Tuple, Final
from numpy.typing import NDArray

# Define type aliases for clarity
Pattern = NDArray[np.int_]
WeightMatrix = NDArray[np.float_]
EnergyList = List[float]

class HopfieldNetwork:
 def __init__(self, num_neurons: int) -> None:
  self.N: Final[int] = num_neurons
  self.T: WeightMatrix = np.zeros((num_neurons, num_neurons), dtype=float)
  self.state: Pattern = np.zeros(num_neurons, dtype=int)
  self.U: NDArray[np.float_] = np.zeros(num_neurons, dtype=float)

 def store_memories(self, memories: List[Pattern], clipped: bool = False) -> None:
  for s in memories:
   bipolar: Pattern = 2 * s - 1 
   delta: WeightMatrix = np.outer(bipolar, bipolar)
   np.fill_diagonal(delta, 0)
   if clipped:
    self.T += np.sign(delta)
   else:
    self.T += delta

 def update_neuron(self, i: int) -> bool:
  old_state: int = self.state[i]
  activation: float = float(np.dot(self.T[i], self.state))
  self.state[i] = int(activation >= self.U[i])
  return old_state != self.state[i]

 def update_async(self, max_iterations: int = 100) -> int:
  for iteration in range(max_iterations):
   i: int = np.random.randint(self.N)
   changed: bool = self.update_neuron(i)
   if not changed:
    return iteration
  return max_iterations

 def update_sync(self, max_iterations: int = 100, tolerance: float = 1e-6) -> int:
  for iteration in range(max_iterations):
   old_state: Pattern = self.state.copy()
   activations: NDArray[np.float_] = np.dot(self.T, self.state)
   self.state = (activations >= self.U).astype(int)
   change_ratio: float = float(np.sum(np.abs(old_state - self.state))) / self.N
   if change_ratio < tolerance:
    return iteration
  return max_iterations

 def energy(self) -> float:
  return -0.5 * float(np.dot(self.state, np.dot(self.T, self.state)))

 def recall(
  self, 
  initial_state: Pattern, 
  max_iterations: int = 100, 
  sync: bool = False
 ) -> Tuple[Pattern, EnergyList]:
  self.state = initial_state.copy()
  energies: EnergyList = [self.energy()]
  iterations: int = self.update_sync(max_iterations) if sync else self.update_async(max_iterations)
  for _ in range(iterations):
   energies.append(self.energy())
  return self.state, energies

 @staticmethod
 def generate_correlated_memories(
  num_memories: int, 
  num_neurons: int, 
  correlation: float
 ) -> List[Pattern]:
  base: Pattern = np.random.choice([0, 1], size=num_neurons).astype(int)
  memories: List[Pattern] = [base.copy()]
  for _ in range(1, num_memories):
   flip: NDArray[np.bool_] = np.random.random(num_neurons) > correlation
   new_mem: Pattern = base.copy()
   new_mem[flip] = 1 - new_mem[flip]
   memories.append(new_mem)
  return memories

 @staticmethod
 def add_noise(pattern: Pattern, noise_level: float) -> Pattern:
  noisy: Pattern = pattern.copy()
  flip: NDArray[np.bool_] = np.random.random(len(pattern)) < noise_level
  noisy[flip] = 1 - noisy[flip]
  return noisy

def test_network(
 N: int = 100, 
 n: int = 10, 
 clipped: bool = False, 
 correlation: float = 0.0, 
 noise_level: float = 0.2, 
 sync: bool = False
) -> None:
 net: HopfieldNetwork = HopfieldNetwork(N)
 memories: List[Pattern] = HopfieldNetwork.generate_correlated_memories(n, N, correlation)
 net.store_memories(memories, clipped)

 for i, mem in enumerate(memories):
  noisy: Pattern = HopfieldNetwork.add_noise(mem, noise_level)
  recalled, energies = net.recall(noisy, sync=sync)
  errors: int = int(np.sum(recalled != mem))
  print(
   f"Memory {i+1}: Errors = {errors}, "
   f"Energy change: {energies[0]:.2f} -> {energies[-1]:.2f}"
  )

def test_familiarity(
 N: int = 100, 
 n: int = 10, 
 thresholds: List[float] = [0.0, 0.1, 0.2, 0.3]
) -> None:
 net: HopfieldNetwork = HopfieldNetwork(N)
 memories: List[Pattern] = HopfieldNetwork.generate_correlated_memories(n, N, correlation=0.0)
 net.store_memories(memories)

 for U in thresholds:
  net.U.fill(U)
  familiar_index: int = np.random.randint(len(memories))
  familiar: Pattern = memories[familiar_index]
  unfamiliar: Pattern = np.random.choice([0, 1], size=N).astype(int)
  
  _, fam_energies = net.recall(familiar)
  _, unfam_energies = net.recall(unfamiliar)
  
  print(
   f"Threshold {U}: Familiar E = {fam_energies[-1]:.2f}, "
   f"Unfamiliar E = {unfam_energies[-1]:.2f}"
  )

if __name__ == "__main__":
 print("Asynchronous update:")
 test_network(clipped=True, correlation=0.2, noise_level=0.1)
 
 print("\nSynchronous update:")
 test_network(clipped=True, correlation=0.2, noise_level=0.1, sync=True)
 
 print("\nFamiliarity test:")
 test_familiarity()
