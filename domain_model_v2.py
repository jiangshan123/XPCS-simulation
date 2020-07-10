import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as colors
import time,gzip

import os

from scipy.signal import convolve2d
from tqdm import tqdm,trange

from numba import jit


"""
General idea here is the electric field, strain, etc matrixes only account for 
the other cells in the lattice, not this one. 
"""

class Model():
    def __init__(self):
        self.assign_parameters()
        self.saving = False
    def assign_parameters(self,**kwargs):
        """
        Note: e0 = 1 in my units
        Distance between unit cells = 1
        free_charges: charge/efield, amount of charge that moves to a surface
                      location based on the local electric field
        landau_4: energy/dipole_moment^4
        landau_6: energy/dipole_moment^6
        temperature: kelvin
        transition_temperature: kelvin
        curie_const: not sure rn
        efield_cost: energy/(efield*dipole_moment)
        dipole_const: unit of our dipoles
        screening_charge: percentage of surface charge that gets screened
        strain_cost: energy/strain, cost of being polarized
        temp_variation: kelvin, the std. dev of temperature variations
        dipole_electric_field: the const in the field from electric dipole 
            other than the moment/r**3
        surface_charge_cost: energy/something. cost of having charge in a field
        """
        #defaults
        self.parameters = {'free_charges': 1, #charge/efield
                           'landau_4': 1,
                           'landau_6': 1,
                           'temperature': 400,
                           'transition_temperature': 420,
                           'curie_const': 1,
                           'efield_cost': 100,
                           'dipole_const': 1,
                           'screening_charge': 0,
                           'strain_cost': 1,
                           'temp_variation': 5,
                           'dipole_electric_field': -3/(4*np.pi),
                           'surface_charge_cost': 5
                          }
        for key,value in kwargs.items():
            self.parameters[key] = value
        self.params = self.parameters
    def init_model(self,shape,moment_delta,dipole_radius,temp_radius):

        self.moment_delta_scale = moment_delta

        self.shape = shape
        
        self.dipole_moments = np.zeros(self.shape)
        self.moment_average = np.zeros(self.shape)
        self.temperature = np.ones(self.shape)*self.params['temperature']
        self.free_energy = np.zeros(self.shape)
        self.moment_delta = np.ones(shape)*moment_delta

        """
        Temp box is just the average of the nearby temperatures
        """
        l = temp_radius * 2 + 1
        self.temperature_box = np.ones((l,l))/l**2

        """
        rbox represents the weighted average of the dipole moments, weighted
        by the distance from the center. the center has 0 weight
        """ 
        l = dipole_radius * 2 + 1
        self.rbox = np.zeros((l,l))
        for i,j in np.ndindex(self.rbox.shape):
            if not i == dipole_radius and not j == dipole_radius:
                r = np.sqrt((i-dipole_radius)**2 + (j-dipole_radius)**2)
                self.rbox[i,j] = 1/r
        self.r2box = self.rbox ** 2
        self.r3box = self.rbox ** 3

        self.r = np.zeros(self.shape)
        self.r2 = np.zeros(self.shape)
        self.r3 = np.zeros(self.shape)
    def random_moments(self):
        self.dipole_moments = np.random.choice([-1,1],size=self.shape)
    def free_energy_func(self,moment,temp,r,r2,r3):
        """
        Moment: dipole moment of this cell
        temp: temperature of this cell
        r,r2,r3: the weighted average of the neighboring dipole moments, 
        weighted by 1/r, 1/r^2, 1/r^3, respectively 

        Intrinsic double well: the landau theory, the unit cell would rather
        be polarized than not polarized (below transition temperature)

        Electric field: the dipole moment would rather follow the electric 
        field than not

        free energy = (t - t0)/(2*C)*P^2 + a4*P^4 + a6*P^6
                      + aE*E*P + aSCC*DP*P
        
        P: dipole moment (polarization)
        t: temperature
        t0: transition temperature
        C: curie constant
        a4,a6: landau parameters
        E: electric field
        aE: energy cost of being polarized in electric field
        DP: depolarization field
        aSCC: energy cost of putting surface charge into a field
        """

        intrinsic_double_well = (temp -self.params['transition_temperature'])*\
                                (1/(2*self.params['curie_const']))*moment**2
        intrinsic_double_well += self.params['landau_4'] * moment ** 4
        intrinsic_double_well += self.params['landau_6'] * moment ** 6

        dipole_electric_field = moment*self.params['dipole_electric_field']*r3*\
                                self.params['efield_cost']

        depolarization_field = self.params['free_charges'] * \
                                  (1-self.params['screening_charge']) * r2
        surface_charge_energy = moment*depolarization_field*\
                                self.params['surface_charge_cost']
        
        strain_energy = self.params['strain_cost']*moment*r

        return intrinsic_double_well + dipole_electric_field + \
               surface_charge_energy + strain_energy
    def update_dipole_moments(self):
        positive = self.free_energy_func(self.dipole_moments + self.moment_delta,
                            self.temperature,self.r,self.r2,self.r3)
        negative = self.free_energy_func(self.dipole_moments - self.moment_delta,
                            self.temperature,self.r,self.r2,self.r3)
        change_to_negative = np.logical_and(negative < positive,
                                        negative < self.free_energy)
        change_to_positive = np.logical_and(positive < negative,
                                        positive < self.free_energy)
        self.dipole_moments = np.where(change_to_negative,
                                        self.dipole_moments-self.moment_delta,
                                        self.dipole_moments)
        self.dipole_moments = np.where(change_to_positive,
                                        self.dipole_moments+self.moment_delta,
                                        self.dipole_moments)
        # self.free_energy = np.where(change_to_negative,
        #                                 negative,self.free_energy)
        # self.free_energy = np.where(change_to_positive,
        #                                 positive,self.free_energy)
        self.free_energy = self.free_energy_func(self.dipole_moments,
                self.temperature,self.r,self.r2,self.r3)
    def update_temperature(self):
        self.temperature = convolve2d(self.temperature,self.temperature_box,\
                                mode='same',boundary='wrap')
        changes = np.random.normal(0,self.params['temp_variation'],self.shape)
        self.temperature += changes
    def init_run(self,fn,num_steps,steps):
        self.saving = True
        self.filename = fn + '.npy.gz'
        self.iteration = 0
        self.step = 0
        self.max_steps = num_steps
        self.steps = steps
        shape = list(self.shape)
        shape.append(num_steps)
        shape = tuple(shape)
        self.data_holder = np.zeros(shape)
    def tick(self):
        #update our convolution boxes
        self.r = convolve2d(self.dipole_moments,self.rbox,
                    mode='same',boundary='wrap')
        self.r2 = convolve2d(self.dipole_moments,self.r2box,
                    mode='same',boundary='wrap')
        self.r3 = convolve2d(self.dipole_moments,self.r3box,
                    mode='same',boundary='wrap')
        #generate a new change in the moments
        self.moment_delta = np.abs(np.random.normal(0,self.moment_delta_scale))
        #update the temperature
        self.update_temperature()

        #update the polarizations
        self.update_dipole_moments()

        #do the saving and whatnot
        if self.saving:
            self.iteration += 1
            if self.iteration%self.steps == 0:
                self.data_holder[:,:,self.step] = self.dipole_moments
                self.step += 1
                if self.step >= self.max_steps:
                    self.saving = False
                    f = gzip.GzipFile(self.filename,'w')
                    np.save(f,self.data_holder)
                    f.close()


class Animator():
    def __init__(self,model):
        self.model = model
        self.fig = plt.figure()
    def animate(self,n):
              
        self.fig.clear()
        plt.subplot(111)
        plt.imshow(model.dipole_moments,cmap='jet',vmin=-.5,vmax=1.5)
        plt.title(n)
        # plt.subplot(212)
        # x = np.linspace(-2,2,num=200)
        # y = self.model.free_energy_func(x,self.model.temperature[0,0],
        #                 self.model.r[0,0],self.model.r2[0,0],self.model.r3[0,0])
        # xx = self.model.dipole_moments[0,0]
        # yy = self.model.free_energy[0,0]
        # plt.imshow(model.temperature,cmap='jet')
        # plt.plot(x,y)
        # plt.scatter([xx],[yy],c='r',s=10)
        for i in trange(50):
            self.model.tick()


shape = (100,100)
delta = 0.01
radius = 5
temp_radius = 3

model = Model()
model.init_model(shape,delta,radius,temp_radius)
np.random.seed(8675309)
model.random_moments()
# model.dipole_moments = np.ones(shape)
# model.dipole_moments[0,0] = -1

animator = Animator(model)

# for i in trange(20):
#     model.tick()

# model.init_run('test',20,50)

ani = animation.FuncAnimation(animator.fig,animator.animate,frames=10000)

plt.show()


        
