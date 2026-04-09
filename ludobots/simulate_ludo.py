import pybullet as p
import time as t

physicsClient = p.connect(p.GUI)


for i in range(1000):
    p.stepSimulation()
    t.sleep(1/60)
    print("Step: ", i)

p.disconnect()